To test/reproduce results, change "option/test/test_ntireEx.json" file in which you need to change path for dataset and pre-trained model of SR network.
(You can find pre-trained network in the main folder with 297000_G.pth file)

Then you need run following command.
python test.py -opt option/test/test_ntire2.json


Required Packages.
pytorch 1.4
opencv 3.4.2
python-lmdb 0.96
