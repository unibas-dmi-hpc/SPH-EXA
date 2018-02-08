# SPH-EXA mini-app

The smooth particle hydrodynamics (SPH) technique is a purely Lagrangian method, used in numerical simulations of fluids in astrophysics and computational fluid dynamics, with no subjacent mesh. SPH simulations represent computationally demanding calculations. Therefore, trade-offs are made between temporal and spatial scales, resolution, dimensionality (3-D or 2-D), and approximated versions of the physics involved. The parallelization of SPH codes is not trivial due to their boundless nature and the absence of a structured particle grid.
[SPHYNX](https://astro.physik.unibas.ch/sphynx/), [ChaNGa](http://faculty.washington.edu/trq/hpcc/tools/changa.html), and [SPH-flow](http://www.sph-flow.com) are the three SPH codes selected in the PASC SPH-EXA project proposal. The performance of these codes is negatively impacted by factors, such as multiple time-stepping and gravity. Therefore, the goal is to extrapolate their common basic SPH features, which are consolidated in a fully optimized, Exascale-ready, MPI+X, pure-SPH, mini-app. The SPH mini-app will integrate further specific physics models.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With
```
* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds
```
## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Danilo Guerrera** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
