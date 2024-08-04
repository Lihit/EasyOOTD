# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : EasyOOTD
# @FileName: setup.py

from setuptools import setup

setup(
    name='easy_ootd',
    version='1.0.0',
    description='EasyOOTD for Virtual TryOn',
    author='wenshao',
    author_email='wenshaoguo1026@gmail.com',
    packages=[
        'easy_ootd',
        'easy_ootd.models',
        'easy_ootd.pipelines',
        'easy_ootd.common',
    ],
    install_requires=[],
    data_files=[]
)
