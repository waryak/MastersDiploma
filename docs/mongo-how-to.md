## How to mongo 

# First, install mongo on ubuntu:
1) sudo apt-get update
2) sudo apt-get install -y mongodb-org

# Download certificates
1) mkdir ~/.mongodb/
2) wget "https://storage.yandexcloud.net/mdb/CA.pem" -O ~/.mongodb/CA.pem

# Connect using shell
```shell
mongo --norc \
        --ssl \
        --sslCAFile ~/.mongodb/CA.pem \
        --host 'rs01/rc1c-iqnqnjr0rs54ud1m.mdb.yandexcloud.net:27018' \
        -u user1 \
        -p <password> \
        db1
```
# Connect from python
1) pip install pyMongo
``` python
from urllib.parse import quote_plus as quote

import ssl
import pymongo

url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
    user=quote('user1'),
    pw=quote('<password>'),
    hosts=','.join([
        'rc1c-iqnqnjr0rs54ud1m.mdb.yandexcloud.net'
    ]),
    rs='rs01',
    auth_src='db1')
dbs = pymongo.MongoClient(
    url,
    ssl_ca_certs='/.mongodb/CA.pem',
    ssl_cert_reqs=ssl.CERT_REQUIRED)['db1']
```
