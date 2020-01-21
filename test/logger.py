from lab import logger
from lab.logger.colors import Text, Color

logger.log("Colors are missing when views on github", Text.highlight)

logger.log([
    ('Styles\n', Text.heading),
    ('Danger\n', Text.danger),
    ('Warning\n', Text.warning),
    ('Meta\n', Text.meta),
    ('Key\n', Text.key),
    ('Meta2\n', Text.meta2),
    ('Title\n', Text.title),
    ('Heading\n', Text.heading),
    ('Value\n', Text.value),
    ('Highlight\n', Text.highlight),
    ('Subtle\n', Text.subtle)
])

logger.log([
    ('Colors\n', Text.heading),
    ('Red\n', Color.red),
    ('Black\n', Color.black),
    ('Blue\n', Color.blue),
    ('Cyan\n', Color.cyan),
    ('Green\n', Color.green),
    ('Orange\n', Color.orange),
    ('Purple Heading\n', [Color.purple, Text.heading]),
    ('White\n', Color.white),
])