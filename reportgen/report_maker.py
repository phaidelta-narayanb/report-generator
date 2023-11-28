
from datetime import datetime, date
from dataclasses import dataclass, field

from jinja2.environment import Environment

from reportgen.utils import get_date_string


@dataclass
class ReportMaker:
    template_env: Environment
    current_date: date = None
    additional_static_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.current_date is None:
            self.current_date = datetime.now().date()

    def from_template(self, report_template: str, **context_args) -> str:
        context = {
            "current_date": get_date_string(),
            **self.additional_static_kwargs,
            **context_args
        }
        return self.template_env.get_template(report_template).render(context)
