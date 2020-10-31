#!/bin/bash

echo "Downloading & Extracting SHREC19 correspondences"
wget https://www.dropbox.com/s/ixz7j9rjdg4nquy/shrec19_corres.tar.bz2
tar xvjf shrec19_corres.tar.bz2
rm -f shrec19_corres.tar.bz2

echo "Downloading & Extracting SHREC19 Partial Scans"
wget https://www.dropbox.com/s/c9xcfn0jq1vv6b5/shrec19_scans.tar.bz2
tar xvjf shrec19_scans.tar.bz2
rm -f shrec19_scans.tar.bz2
