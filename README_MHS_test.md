
## Step to execute all the combinations
**Required**: screen to keep the execution running `sudo apt install screen`
1. Activate screen with `screen` and then press Enter\
Note: to return to the screen other times use `screen -r`
2. Virtual Environment\
2.1 Creation `python -m venv venv`   
2.2 Activation `source venv/bin/activate`\
2.3 Requirements `pip install -r requirements.txt`
3. Launch with `python scores.py`


#
*Note*: 
- Change `logging.set_verbosity_error()` to `logging.set_verbosity_info()` in line 5 of **scores.py** to get a more detailed log.
- Is suggested to launch the script with `python scores.py > scores.txt` to print the output to file 
- To run a single test use `python main.py`, you can change the split settings in row 8 (the default are: user_adaptation =  "train", extended = False and named = True)
- The settings about batch_size e split_percentage are written in `config.py`
