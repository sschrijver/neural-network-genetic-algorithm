# Evolve XGBRegressor with a genetic algorithm

This is an example of how we can use a genetic algorithm in an attempt to find the optimal extreme gradient boosting parameters for classification tasks.

This code is based upon an repository that used an genetic algorithm to evolve a neural network.
That repository can be found [here](https://github.com/harvitronix/neural-network-genetic-algorithm).  

## To run

To run the brute force algorithm:

```python3 brute.py```

To run the genetic algorithm:

```python3 main.py```

You can set your network parameter choices by editing each of those files first. You can also choose whether to use the MNIST or CIFAR10 datasets. Simply set `dataset` to either `mnist` or `cifar10`.

## Credits

The genetic algorithm code is based on the code from this excellent blog post: https://lethain.com/genetic-algorithms-cool-name-damn-simple/

## Contributing

Have an optimization, idea, suggestion, bug report? Pull requests greatly appreciated!

## License

MIT
