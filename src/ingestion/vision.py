import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def get_vision_model() -> ChatOpenAI:
    """
    Create the OpenAI vision model used for image understanding.
    """

    model_name = os.getenv(
        "OPENAI_VISION_MODEL",
        "gpt-4.1-mini",
    )

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )


def describe_image(
    image_base64: str,
) -> str:
    """
    Generate a detailed searchable description
    of an image using an OpenAI vision model.
    """

    if not image_base64:
        return ""

    vision_model = get_vision_model()

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analyze this image for a multimodal "
                    "RAG knowledge base.\n\n"
                    "Describe all useful information visible "
                    "in the image that could help answer "
                    "user questions.\n\n"
                    "Include:\n"
                    "- visible text\n"
                    "- headings\n"
                    "- labels\n"
                    "- numbers\n"
                    "- values\n"
                    "- tables or chart information\n"
                    "- important relationships\n"
                    "- trends or comparisons\n"
                    "- any banking or financial information\n\n"
                    "Do not invent information that is not "
                    "visible in the image.\n"
                    "Return only the detailed description."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64,"
                        f"{image_base64}"
                    )
                },
            },
        ]
    )

    response = vision_model.invoke(
        [message]
    )

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for part in content:

            if (
                isinstance(part, dict)
                and part.get("type") == "text"
            ):

                text_parts.append(
                    part.get("text", "")
                )

        return " ".join(
            text_parts
        ).strip()

    return str(content).strip()


def enrich_image_elements(
    parsed_elements: list[dict],
) -> list[dict]:
    """
    Generate searchable descriptions for all image elements.

    The original image remains in metadata as Base64.
    The generated description becomes the searchable content.
    """

    enriched_elements = []

    for element in parsed_elements:

        if element.get(
            "content_type"
        ) != "image":

            enriched_elements.append(
                element
            )

            continue

        metadata = element.get(
            "metadata",
            {}
        )

        image_base64 = metadata.get(
            "image_base64"
        )

        if not image_base64:

            enriched_elements.append(
                element
            )

            continue

        print()
        print(
            "Generating vision description..."
        )

        try:

            description = describe_image(
                image_base64
            )

            if description:

                enriched_element = {
                    **element,
                    "content": description,
                }

                enriched_elements.append(
                    enriched_element
                )

            else:

                enriched_elements.append(
                    element
                )

        except Exception as exc:

            print(
                "Vision processing failed: "
                f"{exc}"
            )

            # Do not fail the complete ingestion
            # because of one image.
            enriched_elements.append(
                element
            )

    return enriched_elements