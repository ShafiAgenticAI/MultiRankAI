import os
import re
import base64
import io
from pathlib import Path

# Disable PyTorch compilation on Windows.
# This avoids the "cl is not found" compiler issue.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch

torch._dynamo.config.suppress_errors = True

from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

load_dotenv()


def create_converter() -> DocumentConverter:
    """
    Create and configure the Docling PDF converter.
    """

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU
        ),
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        },
    )

    return converter


def parse_document(file_path: Path) -> list[dict]:
    """
    Parse a PDF into multimodal elements.

    Supported content types:
        - text
        - table
        - image

    Each element contains:
        - content
        - content_type
        - metadata
    """

    print("=" * 60)
    print("MULTIMODAL DOCUMENT PARSING STARTED")
    print("=" * 60)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    print(
        f"Processing: {file_path.name}"
    )

    # ---------------------------------------------------------
    # Step 1: Create Docling converter
    # ---------------------------------------------------------

    converter = create_converter()

    # ---------------------------------------------------------
    # Step 2: Convert PDF
    # ---------------------------------------------------------

    result = converter.convert(
        str(file_path)
    )

    doc = result.document

    # ---------------------------------------------------------
    # Step 3: Initialize variables
    # ---------------------------------------------------------

    parsed_chunks: list[dict] = []

    current_section: str | None = None

    source_file = file_path.name

    # ---------------------------------------------------------
    # Step 4: Iterate through document elements
    # ---------------------------------------------------------

    for item in doc.iterate_items():

        if isinstance(item, tuple):
            node, _ = item
        else:
            node = item

        label = str(
            getattr(
                node,
                "label",
                ""
            )
        ).lower()

        # -----------------------------------------------------
        # Skip repeated page headers and footers
        # -----------------------------------------------------

        if label in (
            "page_header",
            "page_footer",
        ):
            continue

        # -----------------------------------------------------
        # Extract page number and position
        # -----------------------------------------------------

        prov = getattr(
            node,
            "prov",
            None
        )

        page_no = None
        position = None

        if prov:

            page_no = prov[0].page_no

            if (
                hasattr(
                    prov[0],
                    "bbox"
                )
                and prov[0].bbox is not None
            ):

                bbox = prov[0].bbox

                position = {
                    "l": bbox.l,
                    "t": bbox.t,
                    "r": bbox.r,
                    "b": bbox.b,
                }

        # -----------------------------------------------------
        # Metadata helper
        # -----------------------------------------------------

        def make_metadata(
            content_type: str,
            element_type: str,
            image_base64: str | None = None,
        ) -> dict:

            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,
                "image_base64": image_base64,
            }

        # -----------------------------------------------------
        # Section headers and document title
        # -----------------------------------------------------

        if (
            "section_header" in label
            or label == "title"
        ):

            text = getattr(
                node,
                "text",
                ""
            ).strip()

            if text:

                current_section = text

                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": make_metadata(
                            "text",
                            label,
                        ),
                    }
                )

        # -----------------------------------------------------
        # Tables
        # -----------------------------------------------------

        elif "table" in label:

            table_text = ""

            if hasattr(
                node,
                "export_to_dataframe"
            ):

                try:

                    df = node.export_to_dataframe(
                        doc=doc
                    )

                    if (
                        df is not None
                        and not df.empty
                    ):

                        # -------------------------------------------------
                        # Docling can return two different table structures:
                        #
                        # 1. Proper column names
                        # 2. Numeric column names [0, 1, 2]
                        #
                        # In the second case, the first row is the
                        # actual table header.
                        # -------------------------------------------------

                        numeric_columns = all(
                            isinstance(
                                column,
                                int
                            )
                            for column in df.columns
                        )

                        if numeric_columns:

                            headers = [
                                str(value).strip()
                                for value in df.iloc[0]
                            ]

                            data_df = df.iloc[1:].copy()

                            data_df.columns = headers

                        else:

                            headers = [
                                str(column).strip()
                                for column in df.columns
                            ]

                            data_df = df.copy()

                        # -------------------------------------------------
                        # Clean table headers
                        # -------------------------------------------------

                        cleaned_headers = []

                        for header in headers:

                            # Docling extracted:
                            # "LoanLoan Amount"
                            #
                            # Correct it to:
                            # "Loan Amount"

                            header = re.sub(
                                r"^LoanLoan\s+Amount$",
                                "Loan Amount",
                                header,
                                flags=re.IGNORECASE,
                            )

                            cleaned_headers.append(
                                header
                            )

                        data_df.columns = (
                            cleaned_headers
                        )

                        # -------------------------------------------------
                        # Convert each row into searchable text
                        # -------------------------------------------------

                        rows_text = []

                        for _, row in data_df.iterrows():

                            pairs = []

                            for (
                                header,
                                value,
                            ) in zip(
                                cleaned_headers,
                                row,
                            ):

                                value_text = str(
                                    value
                                ).strip()

                                if value_text in (
                                    "",
                                    "nan",
                                    "None",
                                ):
                                    continue

                                pairs.append(
                                    f"{header}: {value_text}"
                                )

                            if pairs:

                                rows_text.append(
                                    " | ".join(
                                        pairs
                                    )
                                )

                        table_text = "\n".join(
                            rows_text
                        )

                except Exception as exc:

                    print(
                        "Table DataFrame extraction "
                        f"failed: {exc}"
                    )

            # -------------------------------------------------
            # Fallback to raw text
            # -------------------------------------------------

            if not table_text:

                table_text = getattr(
                    node,
                    "text",
                    ""
                ).strip()

            # -------------------------------------------------
            # Store table
            # -------------------------------------------------

            if table_text:

                parsed_chunks.append(
                    {
                        "content": table_text,
                        "content_type": "table",
                        "metadata": make_metadata(
                            "table",
                            "table",
                        ),
                    }
                )

        # -----------------------------------------------------
        # Images / Pictures / Charts
        # -----------------------------------------------------

        elif (
            "picture" in label
            or "figure" in label
            or label == "chart"
        ):

            image_base64 = None

            caption = getattr(
                node,
                "text",
                ""
            ) or ""

            # -------------------------------------------------
            # Extract image from Docling
            # -------------------------------------------------

            try:

                if hasattr(
                    node,
                    "get_image"
                ):

                    pil_image = node.get_image(
                        doc
                    )

                    if pil_image:

                        buffer = io.BytesIO()

                        pil_image.save(
                            buffer,
                            format="PNG"
                        )

                        image_base64 = (
                            base64.b64encode(
                                buffer.getvalue()
                            ).decode(
                                "utf-8"
                            )
                        )

                # Fallback for older Docling versions.
                if (
                    image_base64 is None
                    and hasattr(
                        node,
                        "image"
                    )
                    and node.image
                ):

                    pil_image = getattr(
                        node.image,
                        "pil_image",
                        None
                    )

                    if pil_image:

                        buffer = io.BytesIO()

                        pil_image.save(
                            buffer,
                            format="PNG"
                        )

                        image_base64 = (
                            base64.b64encode(
                                buffer.getvalue()
                            ).decode(
                                "utf-8"
                            )
                        )

            except Exception as exc:

                print(
                    "Image extraction failed: "
                    f"{exc}"
                )

            # -------------------------------------------------
            # Image content
            #
            # Vision model will be added in the next phase.
            # -------------------------------------------------

            if caption.strip():

                content = caption.strip()

            else:

                content = (
                    f"[Image on page {page_no}]"
                )

            parsed_chunks.append(
                {
                    "content": content,
                    "content_type": "image",
                    "metadata": make_metadata(
                        "image",
                        label,
                        image_base64,
                    ),
                }
            )

        # -----------------------------------------------------
        # Normal text
        # -----------------------------------------------------

        else:

            text = getattr(
                node,
                "text",
                ""
            )

            if text and text.strip():

                parsed_chunks.append(
                    {
                        "content": text.strip(),
                        "content_type": "text",
                        "metadata": make_metadata(
                            "text",
                            label,
                        ),
                    }
                )

    # ---------------------------------------------------------
    # Calculate element counts
    # ---------------------------------------------------------

    text_count = sum(
        1
        for item in parsed_chunks
        if item["content_type"] == "text"
    )

    table_count = sum(
        1
        for item in parsed_chunks
        if item["content_type"] == "table"
    )

    image_count = sum(
        1
        for item in parsed_chunks
        if item["content_type"] == "image"
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("PARSING COMPLETED")
    print("=" * 60)

    print(
        f"Text elements   : {text_count}"
    )

    print(
        f"Table elements  : {table_count}"
    )

    print(
        f"Image elements  : {image_count}"
    )

    print(
        f"Total elements  : {len(parsed_chunks)}"
    )

    print("=" * 60)

    return parsed_chunks