import asyncio
import logging
import json
import traceback
from fastapi import Request
from http import HTTPStatus
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Iterable, List,
                    Optional, Tuple, TypedDict, Union, final)
import uuid
from openai.types.chat import (ChatCompletionContentPartParam,
                               ChatCompletionRole)
from starlette.background import BackgroundTask

from .protocal.openai_protocol import *
from ftllm import llm

class ConversationMessage:
    def __init__(self, role:str, content:str):
      self.role = role
      self.content = content

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

class ChatCompletionStreamResponseWithUsage(BaseModel):
    id: str = Field(default_factory = lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory = lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default = None)

class FastLLmCompletion:
  def __init__(self,
               model_name,
               model):
    self.model_name = model_name
    self.model = model
    self.init_fast_llm_model()
    
  def init_fast_llm_model(self):
    pass
  
  def create_error_response(
          self,
          message: str,
          err_type: str = "BadRequestError",
          status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
      return ErrorResponse(message=message,
                            type=err_type,
                            code=status_code.value)

  async def _check_model(self, request: ChatCompletionRequest):
    if request.model != self.model_name:
      return self.create_error_response(
          message=f"The model `{request.model}` does not exist.",
          err_type="NotFoundError",
          status_code=HTTPStatus.NOT_FOUND)
    else:
      return None

  def _parse_chat_message_content(
      self,
      role: ChatCompletionRole,
      content: Optional[Union[str,
                              Iterable[ChatCompletionContentPartParam]]],
  ) -> Tuple[List[ConversationMessage], List[Awaitable[object]]]:
      if content is None:
          return [], []
      if isinstance(content, str):
          return [ConversationMessage(role=role, content=content)], []

      # 暂时不支持图像输入
      raise NotImplementedError("Complex input not supported yet")

  async def create_chat_completion(
      self, request: ChatCompletionRequest, raw_request: Request
  ) -> Union[ErrorResponse, AsyncGenerator[str, None],
              ChatCompletionResponse, 
              Tuple[AsyncGenerator[str, None], AsyncGenerator]]:
      """Completion API similar to OpenAI's API.

      See https://platform.openai.com/docs/api-reference/chat/create
      for the API specification. This API mimics the OpenAI
      ChatCompletion API.

      NOTE: Currently we do not support the following feature:
          - function_call (Users should implement this by themselves)
      """
      error_check_ret = await self._check_model(request)
      if error_check_ret is not None:
          return error_check_ret
      
      query:str = ""
      if request.prompt:
         request.messages.append({"role": "user", "content": request.prompt})
      try:
          conversation: List[ConversationMessage] = []
          for m in request.messages:
              messages, _ = self._parse_chat_message_content(
                  m["role"], m["content"])

              conversation.extend(messages)

          if len(conversation) == 0:
            raise Exception("Empty msg")
          messages = []
          for msg in conversation:
            messages.append({"role": msg.role, "content": msg.content})

      except Exception as e:
          logging.error("Error in applying chat template from request: %s", e)
          traceback.print_exc()
          return self.create_error_response(str(e))

      request_id = f"fastllm-{self.model_name}-{random_uuid()}"
      
      frequency_penalty = 1.0
      if request.frequency_penalty and request.frequency_penalty != 0.0:
        frequency_penalty = request.frequency_penalty

      max_length = request.max_tokens if request.max_tokens else 8192
      #logging.info(request)
      logging.info(f"fastllm input message: {messages}")
      #logging.info(f"input tokens: {input_token_len}")
      
      input_token_len = self.model.get_input_token_len(messages)

      handle = self.model.launch_stream_response(messages,
                        max_length = max_length, do_sample = True,
                        top_p = request.top_p, top_k = request.top_k, temperature = request.temperature,
                        repeat_penalty = frequency_penalty, one_by_one = True)
      result_generator = self.model.stream_response_handle_async(handle)
      # Streaming response
      if request.stream:
          return (self.chat_completion_stream_generator(
              request, raw_request, result_generator, request_id, input_token_len), 
              BackgroundTask(self.check_disconnect, raw_request, request_id, handle))
      else:
          try:
              return await self.chat_completion_full_generator(
                  request, raw_request, handle, result_generator, request_id, input_token_len)
          except ValueError as e:
              return self.create_error_response(str(e))

  async def check_disconnect(self, raw_request: Request, request_id, handle: int):
    while True:
      if await raw_request.is_disconnected():
        self.model.abort_handle(handle)
        logging.info(f"Abort request: {request_id}")
        return
      await asyncio.sleep(1)  # 检查间隔
      
  async def chat_completion_full_generator(
              self, request: ChatCompletionRequest, raw_request: Request,
              handle: int,
              result_generator: AsyncIterator,
              request_id: str,
              input_token_len: int) -> Union[ErrorResponse, ChatCompletionResponse]:
      model_name = self.model_name
      created_time = int(time.time())
      result = ""
      completion_tokens = 0
      async for res in result_generator:
        result += res
        completion_tokens += 1
        if await raw_request.is_disconnected():
           self.model.abort_handle(handle)
           logging.info(f"Abort request: {request_id}")
           return self.create_error_response("Client disconnected")

      choice_data = ChatCompletionResponseChoice(
              index=0,
              message=ChatMessage(role="assistant", content=result),
              logprobs=None,
              finish_reason='stop',
          )

      # TODO: 补充usage信息, 包括prompt token数, 生成的tokens数量
      response = ChatCompletionResponse(
          id = request_id,
          created = created_time,
          model = model_name,
          choices = [choice_data],
          usage = UsageInfo(prompt_tokens = input_token_len,
                            total_tokens = input_token_len + completion_tokens,
                            completion_tokens = completion_tokens)
      )

      return response
      
            
  async def chat_completion_stream_generator(
          self, request: ChatCompletionRequest, raw_request: Request,
          result_generator: AsyncIterator,
          request_id: str,
          input_token_len: int) -> AsyncGenerator[str, None]:
      model_name = self.model_name
      created_time = int(time.time())
      chunk_object_type = "chat.completion.chunk"

      # TODO: 支持request.n 和 request.echo配置
      first_iteration = True
      try:
        if first_iteration:
            # 1. role部分
            choice_data = ChatCompletionResponseStreamChoice(
                            index = 0,
                            delta = DeltaMessage(role = "assistant"),
                            logprobs = None,
                            finish_reason = None)
            chunk = ChatCompletionStreamResponseWithUsage(
                id = request_id,
                object = chunk_object_type,
                created = created_time,
                choices = [choice_data],
                model = model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            first_iteration = False
        
        # 2. content部分
        completion_tokens = 0
        async for res in result_generator:
            completion_tokens += 1
            delta_text = res            
            # Send token-by-token response for each request.n
            choice_data = ChatCompletionResponseStreamChoice(
                index = 0,
                delta = DeltaMessage(content = delta_text),
                logprobs = None,
                finish_reason = None)
            chunk = ChatCompletionStreamResponseWithUsage(
                id = request_id,
                object = chunk_object_type,
                created = created_time,
                choices = [choice_data],
                model = model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            #await asyncio.sleep(0)

        # 3. 结束标志
        choice_data = ChatCompletionResponseStreamChoice(
            index = 0,
            delta = DeltaMessage(),
            logprobs = None,
            finish_reason = 'stop')
        chunk = ChatCompletionStreamResponseWithUsage(
            id = request_id,
            object = chunk_object_type,
            created = created_time,
            choices = [choice_data],
            model = model_name,
            usage = UsageInfo(prompt_tokens = input_token_len,
                              total_tokens = input_token_len + completion_tokens,
                              completion_tokens = completion_tokens))
        data = chunk.model_dump_json(exclude_unset = True,
                                    exclude_none = True)
        yield f"data: {data}\n\n"
      except ValueError as e:
        data = self.create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
        await asyncio.sleep(0)
      yield "data: [DONE]\n\n"
      await asyncio.sleep(0)
      
  def create_streaming_error_response(
          self,
          message: str,
          err_type: str = "BadRequestError",
          status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
      json_str = json.dumps({
          "error":
          self.create_error_response(message=message,
                                      err_type=err_type,
                                      status_code=status_code).model_dump()
      })
      return json_str
    