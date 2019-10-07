from urllib.parse import quote_plus as quote
import ssl
import pymongo

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class MongoManager:
    def __init__(self):
        '''Initiate mongoDB connection
        TODO: get rid of hard coded params
        Make a config with params'''
        url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
            user=quote('user1'),
            pw=quote('diploma_test'),
            hosts=','.join([
                'rc1c-fa410ox5e6dab6yw.mdb.yandexcloud.net:27018'
            ]),
            rs='rs01',
            auth_src='db1')
        self.dbs = pymongo.MongoClient(
            url,
            ssl_ca_certs='/usr/local/share/ca-certificates/Yandex/YandexInternalRootCA.crt',
            ssl_cert_reqs=ssl.CERT_REQUIRED)['db1']
    
    def get_test_lorenz(self):
        '''Get lorenz28 series from test column'''
        col = self.dbs['lorenz28']
        return col.find_one()
    
    def get_data(self,column_name,field_name=None):
        '''Get data using predicate filter'''
        col = self.dbs[column_name]
        if field_name != None:
            return col.find({field_name:{"$exists": True}})
        else:
            return col.find()
    
    def insert_document(self,column_name,document):
        '''insert any JSON-serializable document
        params:
        column_name - string
        document - JSON serializable dict
        '''
        self.dbs[column_name].insert_one(document)