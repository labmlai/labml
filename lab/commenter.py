from typing import List


class Commenter:
    """
    # This class adds header comments to python source files
    """

    def __init__(self, *,
                 comment_start: str,
                 comment_end: str,
                 add_start: str,
                 add_end: str):
        self.comment_start = comment_start
        self.comment_end = comment_end
        self.add_start = add_start
        self.add_end = add_end

    def _trim(self, lines: List[str]):
        """
        ### Remove empty lines
        """

        if len(lines) == 0:
            return

        while lines[0].strip() == "":
            lines.pop(0)

        while lines[-1].strip() == "":
            lines.pop(-1)

    def _extract(self, lines, start: str, end: str):
        """
        ### Extract lines of code between `start` and `end` tokens
        """

        for i, line in enumerate(lines):
            if line.strip() == start:
                return self._extract_from(lines, start, end, i)

        return []

    def _extract_from(self, lines, start: str, end: str, idx: int):
        """
        ### Extract lines of code between `start` and `end` tokens

        This extracts starting for line number `idx`

        🛑 This will just fail, if lines is empty.
        """

        extracted = []

        assert lines[idx].strip() == start
        lines.pop(idx)
        while lines[idx].strip() != end:
            extracted.append(lines[idx])
            lines.pop(idx)

        lines.pop(idx)

        return extracted

    def update(self, code: List[str], add: List[str]):
        """
        ## Update `code` and add lines `add`
        """

        self._trim(code)
        self._trim(add)

        if code[0].strip() == self.comment_start:
            comment = self._extract_from(code, self.comment_start, self.comment_end, 0)
        else:
            comment = []

        self._trim(comment)
        self._trim(code)

        _ = self._extract(comment, self.add_start, self.add_end)

        self._trim(comment)

        if len(comment) > 0:
            comment.append('')

        return ([self.comment_start] +
                comment +
                [self.add_start] +
                add +
                [self.add_end, self.comment_end, ''] +
                code +
                [''])
