#!/usr/bin/env bash

python figures.py

xelatex main.tex
biber main.tex
xelatex main.tex
xelatex main.tex
