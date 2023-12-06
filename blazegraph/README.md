# load and start blazgraph
## first start the docker container
see [here](https://github.com/dsghrg/mt_rdf/blob/main/blazegraph/docker/README.md)

## inside the container run the following commands

### first load data

```bash
./load_data.sh <dataset_name>
```

### start blazegraph server

```bash
./blazegraph_service.sh start <dataset_name>
```

### stop blazegraph server

```bash
./blazegraph_service.sh stop
```