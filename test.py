#%%
import requests
from pprint import PrettyPrinter
 
pp = PrettyPrinter()
pp.pprint(requests.get("http://localhost:8080/v1/models/1709760128").json())


