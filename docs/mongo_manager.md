# How to use mongo_manager.py
1) Create a single instance inst = MongoManager()
2) Use get_data(self,column_name,field_name=None) to get smth. If field_name != None checks that given field exists. Returns iterator.
'''for i in instance.get_data('lorenz28'):
    print(i)'''
3) Use insert_document(self,column_name,document) to insert smth. Document - json serializable dict
