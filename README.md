# Transformers Triple-S
Transformers Triple-S is semantic search system, that uses transformer's powers of text understanding to provide You a comfortable document search.


<img src="https://i.scdn.co/image/ab67616d0000b273b12877d8bdfaa0f19b4624fa" Title="ValhalaHatesOstis">

This is [Colab](https://colab.research.google.com/drive/1YA6n8_EkFoXuT37fpImykPWkEBwNJp5_?usp=sharing) to train transformer. 

To make virtual environment run commands:
```
python3 -m venv venv
```
```
source ./venv/bin/activate
```
```
pip install -r requirements.txt
```
```
pip install -r torch_requirements.txt
```

To run database run commands:
```
docker pull postgres:17
```

```
docker pull pgvector/pgvector:pg17
```

```
docker run -d -v ./postgres_volume:/var/lib/postgresql/data -p 5432:5432 --name triple-s-db -e POSTGRES_PASSWORD=ValhalaWithZolinks pgvector/pgvector:pg17
```
```
python3 ./backend/embedding_system/make_db.py
```
