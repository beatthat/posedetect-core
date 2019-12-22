posedetect-core
===================

Core framework and for the pose detection. Assumes an external implementation implements the actual pose detection by implementing interfaces.

Python Installation
-------------------

```
pip install --user -e git+https://github.com/beatthat/posedetect-core.git@{release-tag}#egg=posedetect_core
```

Creating and Using an implementation
------------------------------------



Exportable Tests
----------------

This package comes with some exportable tests you can run against your implementation.



Development
-----------

Run tests during development with

```
make test-all
```

Once ready to release, create a release tag, currently using semver-ish numbering, e.g. `1.0.0(-alpha.1)`