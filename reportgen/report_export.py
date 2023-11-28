
import tempfile
from dataclasses import dataclass
from typing import Iterable, Literal

@dataclass
class ReportExporter:
    method: Literal["pandoc", "weasyprint"] = "pandoc"
    input_format: str = "md"
    export_args: Iterable = ()
    cache_dir: str = None

    def make_pdf(self, input_data: str, prefix: str="Report_download_") -> str:
        pdf_file_path = tempfile.mktemp(prefix=prefix, suffix=".pdf", dir=self.cache_dir)

        if self.method == "weasyprint":
            from weasyprint import HTML
            HTML(string=input_data).write_pdf(pdf_file_path)

        elif self.method == "pandoc":
            import pypandoc
            pypandoc.ensure_pandoc_installed()

            pypandoc.convert_text(
                input_data,
                "pdf",
                format=self._format,
                outputfile=pdf_file_path,
                extra_args=self._export_args
            )

        return pdf_file_path
