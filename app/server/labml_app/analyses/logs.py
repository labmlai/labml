from enum import Enum

from labml_app.settings import LOG_CHAR_LIMIT

from typing import Any, Dict, List

from labml_db import Key, Model
from labml_db.serializer.pickle import PickleSerializer

from labml_app.analyses.analysis import Analysis


class LogPage:
    logs: str
    logs_unmerged: str

    @classmethod
    def defaults(cls):
        return dict(logs='', logs_unmerged='')

    def update_logs(self, new_logs):
        unmerged = self.logs_unmerged + new_logs
        processed = ''
        if len(new_logs) > 1:
            processed, unmerged = self._format_output(unmerged)

        self.logs_unmerged = unmerged
        self.logs += processed

    @staticmethod
    def _format_output(output: str) -> (str, str):
        res = []
        temp = ''
        for i, c in enumerate(output):
            if c == '\n':
                temp += '\n'
                res.append(temp)
                temp = ''
            elif c == '\r' and len(output) > i + 1 and output[i + 1] == '\n':
                pass
            elif c == '\r':
                temp = ''
            else:
                temp += c

        return ''.join(res), temp

    def get_data(self) -> Dict[str, Any]:
        return {
            'logs': self.logs + self.logs_unmerged,
        }

    def is_full(self):
        return len(self.logs) > LOG_CHAR_LIMIT


@Analysis.db_model(PickleSerializer, 'log_page')
class LogPageModel(Model['LogPageModel'], LogPage):
    pass


class LogPageType(Enum):
    LAST = -1
    ALL = -2


class Logs:
    log_pages: List[Key['LogPageModel']]
    wrap_logs: bool

    @classmethod
    def defaults(cls):
        return dict(
            log_pages=[],
            wrap_logs=True
        )

    def get_data(self, page_no: int = LogPageType.LAST.value):
        page_dict: Dict[str, str] = {}

        if page_no == LogPageType.ALL.value:
            pages: List['LogPage'] = [page.load() for page in self.log_pages]
            for i, p in enumerate(pages):
                page_dict[str(i)] = p.logs + p.logs_unmerged
        elif len(self.log_pages) > page_no >= 0:
            page = self.log_pages[page_no].load()
            page_dict[str(page_no)] = page.logs + page.logs_unmerged

        if len(self.log_pages) > 0:  # always include the last page
            page = self.log_pages[-1].load()
            page_dict[str(len(self.log_pages) - 1)] = page.logs + page.logs_unmerged

        return {
            'pages': page_dict,
            'page_length': len(self.log_pages),
            'wrap_logs': self.wrap_logs,
        }

    def update_opt(self, data: Dict[str, Any]):
        self.wrap_logs = data.get('wrap_logs', True)

    def update_logs(self, content: str):
        if len(self.log_pages) == 0:
            page = LogPageModel()
            page.save()
            self.log_pages.append(page.key)
        else:
            page = self.log_pages[-1].load()

        if page.is_full():
            unmerged_logs = page.logs_unmerged
            page.logs_unmerged = ''
            page.save()
            content = unmerged_logs + content

            page = LogPageModel()
            page.update_logs(content)
            page.save()
            self.log_pages.append(page.key)
        else:
            page.update_logs(content)
            page.save()

    def update_logs_bulk(self, content: str):
        loaded_pages = []
        if len(self.log_pages) == 0:
            page = LogPageModel()
            self.log_pages.append(page.key)
            loaded_pages.append(page)
        else:
            page = self.log_pages[-1].load()

        for line in content.split('\n'):
            line = line + "\n"
            if page.is_full():
                unmerged_logs = page.logs_unmerged
                page.logs_unmerged = ''
                line = unmerged_logs + line

                page = LogPageModel()
                page.update_logs(line)
                self.log_pages.append(page.key)
                loaded_pages.append(page)
            else:
                page.update_logs(line)

        LogPageModel.msave(loaded_pages)
