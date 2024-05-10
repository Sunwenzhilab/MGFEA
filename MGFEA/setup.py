#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: luojiaxu
# Mail: 925009185@qq.com
# Created Time:  2024-5-9 
#############################################

from setuptools import setup, find_packages            

setup(
    name = "MGFEA",      
    version = "0.0.1",  
    keywords = ["pip", "MGFEA"],
    description = "A package for single cell flux prediction",
    long_description = "A deep learning framework for metabolic flux prediction based on single cell transcriptome",
    license = "MIT Licence",

    url = "https://github.com/sunwenzhi-cibr/MGFEA/",     
    author = "luojiaxu",
    author_email = "925009185@qq.com",

    packages = find_packages(include=['MGFEA']),

    install_requires = ['torch','scipy','scanpy']          
)