from setuptools import setup

setup(
    use_scm_version={
        "write_to": "_version.py",
        "write_to_template": 'version = "{version}"\n',
    }
)
