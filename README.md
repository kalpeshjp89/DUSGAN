# Direct Unsupervised Super-Resolution using Generative Adversarial Network (DUS-GAN) for Real-World Data

The repository contains the official code for the work **"Direct Unsupervised Super-Resolution using Generative Adversarial Network (DUS-GAN) for Real-World Data"**.

**- Pre-Trained models**

The pre-trained model for is shared with the repository..

**- Training the model**

Training code has been released. To train the network, run the following command.
```javascript
python train.py -opt path_for_training_json_file
```
Note the following changes are needed to run the code.
- Need to provide pre-train QA network path at line number 307 for model/DS_Model.py file.
- Change the root folder and training dataset path into train_ntireEx.json file located at options/train folder.

**- Testng the model**

To test your/our pre-trained model, you need to set root directory and dataset directory into `options/test/test_ntire_ex.json` file. Then run the following command to start the training.
```javascript
python test.py -opt PATH-to-json_file

```

**- Requirement of packages**

The list of packages required to run the code is given in `chasnet.yml` file.

We are thankful to Xinntao for their ESRGAN code using which this work has been implemented.
For any problem, you may contact at <kalpesh.jp89@gmail.com>.
