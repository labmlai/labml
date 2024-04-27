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
        return len(self.logs) > 10


@Analysis.db_model(PickleSerializer, 'log_page')
class LogPageModel(Model['LogPageModel'], LogPage):
    pass


class Logs:
    log_pages: List[Key['LogPageModel']]

    @classmethod
    def defaults(cls):
        return dict(
            log_pages=[]
        )

    def get_data(self, page_no: int = -1):
        logs: str = ''
        if len(self.log_pages) < page_no:
            logs = ''
        elif page_no == -1:
            pages: List['LogPage'] = [page.load() for page in self.log_pages]
            for p in pages:
                logs += p.logs
            logs += pages[-1].logs_unmerged if len(pages) != 0 else ''
        else:
            page = self.log_pages[page_no].load()
            logs = page.logs + page.logs_unmerged
        return {
            'logs': logs
        }

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
            content += unmerged_logs

            page = LogPageModel()
            page.update_logs(content)
            page.save()
            self.log_pages.append(page.key)
        else:
            page.update_logs(content)
            page.save()
