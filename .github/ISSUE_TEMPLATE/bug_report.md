---
name: Bug report
about: Report something that's not working as documented
title: "[BUG] "
labels: bug
---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
Minimal code snippet that triggers the issue:

```python
from hni_vlm import HNIClassifier
model = HNIClassifier(backend="qwen")
# ...
```

**Expected behavior**
What did you expect to happen?

**Actual behavior / error message**
Paste the full traceback if any.

**Environment**
 - OS:               [e.g. Windows 11 / macOS 14 / Ubuntu 22.04]
 - Python version:   [e.g. 3.11.5]
 - hni-vlm version:  [`pip show hni-vlm`]
 - Backend / model:  [e.g. qwen3-omni-flash]

**Additional context**
Anything else that might help us reproduce or fix the issue.
