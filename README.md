# Welcome 👋

This is a monorepo containing various survival analysis projects.
The main focus is on federated deep survival analysis.

# Run projects 🚀

We use PDM as a dependency manager. Additional steps can be found in project-level READMEs.

To install each project run:

```sh
pdm use 3.8 && pdm install
```

Tu run scripts that you would run with `python3` use `pdm run` instead:

```sh
pdm run <your command>
```

> Examples:
> ```sh
> pdm run main.py
> ```
>
> ```sh
> pdm run jupyter kernelspec list
> ```

To run modules (e.g. scripts that you would run with `python3 -m`) use `pdm run python3 -m` instead:

```sh
pdm run python3 -m <your command>
```

> Example:
> ```sh
> pdm run python3 -m ipykernel install --help
> ```

# Run Jupyter notebooks 📓

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

🐈
