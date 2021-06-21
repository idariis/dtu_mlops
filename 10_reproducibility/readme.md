# 10. Reproducibility and configuration files

Today is all about reproducibility - one of those concepts that everyone agrees is very important and something
should be done about, but reality is that it is very hard to secure full reproducibility.

For the first set of exercises we have provided a single script. Your task is to use Hydra to make sure that everything
gets correctly logged such that you would be able to exactly report to others how each experiment was configured. In the
provided script, the hyperparameters are hardcoded into the code and your job will be to separate them out into a 
configuration file.

### Exercises

Note that we provide an solution (in the `vae_solution` folder) that can help you get through the exercise, 
but try to look online for your answers before looking at the solution. Remember: its not about the result, its about the journey.

1. Start by install hydra: `pip install hydra-core --upgrade`

2. Next take a look at the `vae_mnist.py` and `model.py` file and understand what is going on. It is the same model 
   that you debugged in part 3 (just without the bugs).
   
3. Identify the key hyperparameters of the script. Some of them should be easy to find, but at least 3 have
   made it into the core part of the code. One essential hyperparameter is also not included in the script
   but is needed to be completely reproducible (HINT: the weights of any neural network is initialized at
   random).
   
4. Write a configuration file `config.yaml` where you write down the hyparameters that you have found

5. Get the script running by loading the configuration file inside your script (using hydra) that incorporates
   the hyperparameters into the script. Note: you should only edit the `vae_mnist.py` file and not the `model.py` file.
   
6. Run the script

7. By default hydra will write the results to a `outputs` folder, with a sub-folder for the day the experiment
   was run and further the time it was started. Inspect your run by going over each file the hydra has generated
   and check the information has been logged. Can you find the hyperparameters?
   
8. Hydra also allows for dynamically changing and adding parameters on the fly from the command-line:

   8.1 Try changing one parameter from the command-line
       ```
       python vae_mnist.py seed=1234
       ```

   8.2 Try adding one parameter from the command-line
       ```
       python vae_mnist.py +experiment.stuff_that_i_want_to_add=42
       ```

9. By default the file `vae_mnist.log` should be empty, meaning that whatever you printed to the terminal
   did not get picked up by Hydra. This is due to Hydra under the hood making use of the native python 
   [logging](https://docs.python.org/3/library/logging.html) package. This means that to also save all 
   printed output from the script we need to convert all calls to `print` with `log.info`

   8.1 Create a logger in the script:
       ```python
       import logging
       log = logging.getLogger(__name__)
       ```

   8.2 Exchange all calls to `print` with calls to `log.info`

   8.3 Try re-running the script and make sure that the output printed to the terminal also gets saved to the
       `vae_mnist.log` file

10. Make sure that your script is fully reproducible. To check this you will need two runs of the script to
    compare. Then run the `reproduceability_tester.py` script as
    ```
    python reproducibility_tester.py path/to/run/1 path/to/run/2
    ```
    the script will go over trained weights to see if the match and that the hyperparameters was the same. Note:
    for the script to work, the weights should be saved to a file called `trained_model.pt` (this is the default
    of the `vae_mnist.py` script, so only relevant if you have changed the saving of the weights)

11. Finally, make a new experiment using a new configuration file where you have changed a hyperparameter of
    your own choice. You are not allowed to change the configuration file in the script but should instead
    be able to provide it as an argument when launching the script e.g. something like
    ```python vae_mnist.py experiment=exp2```
  
### Final exercise

Make your MNIST code reproducible! Apply what you have just done to the simple script to your MNIST code.
Only requirement is that you this time use multiple configuration files, meaning that you should have at least
one `model_conf.yaml` file and a `training_conf.yaml` file that separates out the hyperparameters that have to
do with the model definition and those that have to do with the training.
