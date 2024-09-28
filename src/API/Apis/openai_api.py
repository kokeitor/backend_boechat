from openai import OpenAI
from decouple import config
import openai


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
            {"role": "system", "content":  "Eres un asistente personal."}
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
                "role": "system",
                "content": completeMessage.choices[0].message.content
            }
        )
        return self.messages[-1]["content"]

    def getStreamResponse(self, newUserMessage: str):
        self.messages.append(
            {
                "role": "user",
                "content": newUserMessage
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
            print("event.choices[0].delta.content : ",
                  event.choices[0].delta.content)
            if event.choices[0].delta.content != None:
                current_response = event.choices[0].delta.content
                completeMessage += current_response
                yield current_response + "\n\n"
                self.completeMessages.append(streamMessage)
        # append the commplete ia response to the memory
        self.messages.append(
            {
                "role": "system",
                "content": completeMessage
            }
        )
        print("complete message : ", completeMessage)
