from openai import OpenAI
from decouple import config
import openai
import logging
from src.API.models.models import OpenAIChatGraph

# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("open_ai_logger")


def getOpenAiClient(OPENAI_API_KEY):
    return OpenAI(
        api_key=OPENAI_API_KEY
    )


class OpenAiModel:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_ky: str = config("OPENAI_API_KEY")
    ) -> None:
        self.client = getOpenAiClient(api_ky)
        self.messages: list[dict[str, str]] = [
            {"role": "assistant", "content":  """"
             Eres un asistente personal que debe responder a una pregunta de un usuario : 'userMesage',\n
             teniendo en cuenta la generacion de otro modelo LLM : 'generation' y el contexto que este modelo a recibido para responder 'context'.\n
             Las preguntas son sobre el Boletin Oficial del Estado Español (BOE). Si en el 'context' o en la 'generation' no se encuentra la respuesta\n
             correcta, debes buscarla tu mismo por tu propias fuentes y contestar correctacmente al usuario"""}
        ]
        self.completeMessages: list[str] = []
        self.files: list = []
        self.model = model
        self.temperature: float = 0

    @staticmethod
    def getOpenAiClient(OPENAI_API_KEY):
        return OpenAI(
            api_key=OPENAI_API_KEY
        )

    def getResponse(self, newUserMessage: str) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": newUserMessage
            }
        )
        completeMessage = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        self.completeMessages.append(completeMessage)
        self.messages.append(
            {
                "role": "assistant",
                "content": completeMessage.choices[0].message.content
            }
        )
        return self.messages[-1]["content"]

    def getResponseFromGraph(self, input: OpenAIChatGraph) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": f"'userMesage' : '{input.userMessage}'\n'context' : '{input.context}'\n'generation' : '{input.generationGraph}'",

            }
        )
        completeMessage = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )

        print(f"\n userMesage : {input.userMessage}")
        print(f"\ncontext : {input.context}")
        print(f"\ngenerationGraph : {input.generationGraph}")
        print(f"\nOpenAi complete response obj : {completeMessage}")
        print(
            f"\nOpenAi response : {completeMessage.choices[0].message.content}")

        logger.info(f"userMesage : {input.userMessage}")
        logger.info(f"context : {input.context}")
        logger.info(f"generationGraph : {input.generationGraph}")
        logger.info(f"OpenAi complete response obj : {completeMessage}")
        logger.info(
            f"OpenAi response : {completeMessage.choices[0].message.content}")

        self.completeMessages.append(completeMessage)
        self.messages.append(
            {
                "role": "assistant",
                "content": completeMessage.choices[0].message.content
            }
        )
        return self.messages[-1]["content"]

    def getStreamResponseFromGraph(self, input: OpenAIChatGraph):
        self.messages.append(
            {
                "role": "user",
                "content": f"'userMesage' : '{input.userMessage}'\n'context' : '{input.context}'\n'generation' : '{input.generationGraph}'",

            }
        )
        streamMessage = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            stream=True
        )
        completeMessage = ""

        # Create the iterator
        for event in streamMessage:
            logger.info(
                f"event.choices[0].delta.content : {event.choices[0].delta.content}")
            if event.choices[0].delta.content != None:
                current_response = event.choices[0].delta.content
                completeMessage += current_response
                logger.info(
                    f"event.choices[0].delta.content : {current_response}")
                yield current_response + "\n\n"
                self.completeMessages.append(streamMessage)

        # append the commplete ia response to the memory
        self.messages.append(
            {
                "role": "assistant",
                "content": completeMessage
            }
        )

        logger.info(f"complete message :  {completeMessage}")
        print(f"userMesage : {input.userMessage}")
        print(f"context : {input.context}")
        print(f"generationGraph : {input.generationGraph}")
        print(f"OpenAi streamresponse : {completeMessage}")

    def getStreamResponse(self, userMessage: str):
        self.messages.append(
            {
                "role": "user",
                "content": userMessage,

            }
        )
        streamMessage = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            stream=True
        )
        completeMessage = ""

        # Create the iterator
        for event in streamMessage:
            logger.info(
                f"event.choices[0].delta.content : {event.choices[0].delta.content}")
            if event.choices[0].delta.content != None:
                current_response = event.choices[0].delta.content
                completeMessage += current_response
                logger.info(
                    f"event.choices[0].delta.content : {current_response}")
                yield current_response + "\n\n"
                self.completeMessages.append(streamMessage)

        # append the commplete ia response to the memory
        self.messages.append(
            {
                "role": "assistant",
                "content": completeMessage
            }
        )

        logger.info(f"complete message :  {completeMessage}")
