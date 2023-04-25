To install a PDM virtual environment as a Jupyter kernel for the current user run:

```sh
pdm run python3 -m ipykernel install --name "<desired kernel id>" --display-name "<desired display name>" --user
```

To list available kernels run:

```sh
pdm run jupyter kernelspec list
```

To uninstall your custom kernel run:

```sh
pdm run jupyter kernelspec uninstall "<kernel id>"
```

