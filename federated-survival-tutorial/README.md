To globally install this virtual environment as a Jupyter kernel run:

```sh
pdm run python3 -m ipykernel install --name my-python-env --display-name "Custom Python environment" --user
```
To list available kernels run:

```sh
pdm run jupyter kernelspec list
```

To uninstall your custom kernel run: 

```sh
pdm run jupyter kernelspec uninstall <your kernel's id>
```

