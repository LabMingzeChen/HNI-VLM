# Contributing to HNI-VLM

Thanks for your interest in contributing! HNI-VLM welcomes contributions of all sizes — from typo fixes to new VLM backends.

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/HNI-VLM.git
cd HNI-VLM
pip install -e ".[dev]"
```

## What we welcome

- 🐛 **Bug fixes** — open an issue first, then submit a PR referencing it.
- 🔌 **New VLM backends** — subclass `BaseVLM` and add tests. GPT-4, Gemini, Claude, LLaVA, InternVL are all welcome.
- 📝 **Documentation improvements** — clearer examples, better docstrings, translations.
- 🧪 **More tests** — the eval pipeline always benefits from more coverage.
- 📊 **Benchmark contributions** — if you run HNI-VLM on a new dataset and want to share results, we'd love to add them to the README table.

## Pull request checklist

- [ ] Code follows the existing style (black + ruff).
- [ ] New code has docstrings.
- [ ] Tests pass (`pytest`).
- [ ] README / docs updated if the public API changed.
- [ ] PR description explains *why*, not just *what*.

## Adding a new backend

1. Create `hni_vlm/models/<your_model>_backend.py`.
2. Subclass `BaseVLM` and implement `_call(self, image_url: str, prompt: str) -> str`.
3. Register it in `hni_vlm/models/__init__.py` and (optionally) in `HNIClassifier.__init__`.
4. Add a usage snippet in the README under "Add a new VLM backend".

## Code of conduct

Be kind. Disagreements about technical decisions are welcome; disrespect is not.

## License

By contributing, you agree that your contributions will be released under the MIT License.
