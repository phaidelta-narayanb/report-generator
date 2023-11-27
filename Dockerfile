FROM python:3.10

RUN apt-get update && apt-get install -y parallel pandoc texlive-latex-base texlive-latex-recommended texlive-fonts-recommended
RUN python3 -m pip install -q --upgrade pip

WORKDIR /app

COPY . .

RUN python3 -m pip install -q -e git+https://github.com/haotian-liu/LLaVA.git@8467850a63aa0d6f47aa150c53aca4751f0d3d14#egg=llava
RUN python3 -m pip install -q --upgrade -r requirements.txt

CMD ["/usr/bin/bash", "run.sh"]
