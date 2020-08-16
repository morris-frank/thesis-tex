#!/usr/bin/env bash

python figures.py

xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
