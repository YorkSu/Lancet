# -*- coding: utf-8 -*-
"""Logic

  File: 
    /Lancet/core/ops/logic

  Description: 
    Logic Operations
"""


from Lancet.core.lib import log


def assert_type(value, t, name=''):
  """Assert Type
  
    Description:
      FIXME
  """
  if isinstance(t, (list, tuple)):
    t = tuple(t)
  else:
    t = tuple([t])
  if not isinstance(value, t):
    # log.error(f"[TypeError] Value is expected as {t}, got "
    #     f"{type(value)}, value: {value}", exit=True, name=__name__)
    name = name or 'Value'
    print(f"[TypeError] `{name}` is expected as {t}, got "
        f"{type(value)}, {name}: {value}")





