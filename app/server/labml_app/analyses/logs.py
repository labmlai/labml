from typing import Any, Dict


class Logs:
    logs: str
    logs_unmerged: str

    @classmethod
    def defaults(cls):
        return dict()

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
