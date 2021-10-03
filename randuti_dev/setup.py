"""setup.py
Install the randuti package to run the test and example codes.

cd into `randuti_dev` and issue

```batch
pip install -e .
```

This will install a file
"%USERPROFILE%/miniconda3/Lib/site-packages/randuti.egg-link"
that will python where the source code should be located,
and also tells that the package information should be 2 levels up.

Reasons to use this technique:

- Results in the simplest code
- This the clean way to include a folder not in your path:
  https://stackoverflow.com/a/50194143/1837006
- The src layer method is the suggested way to create pytest:
  https://docs.pytest.org/en/stable/goodpractices.html#tests-outside-application-code
  https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure

To uninstall, issue

```batch
pip uninstall randuti
```
"""

import setuptools

setuptools.setup()
