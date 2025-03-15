# Transformers Triple-S
Transformers Triple-S is semantic search system, that uses transformer's powers of text understanding to provide You a comfortable document search.


<img src="https://i.scdn.co/image/ab67616d0000b273b12877d8bdfaa0f19b4624fa" Title="ValhalaHatesOstis">

To run database use commands:
```
docker pull postgres:17
```

```
docker pull pgvector/pgvector:pg17
```

```
docker run -d -v ./postgres_volume:/var/lib/postgresql/data -p 5432:5432 --name triple-s-db -e POSTGRES_PASSWORD=ValhalaWithZolinks postgres
```
