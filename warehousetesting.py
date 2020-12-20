# %% 
from CleanDrones import Product, Order, Shipment, Warehouse, calculate_redistribution
import unittest
import numpy as np

# %%

class test_warehouses(unittest.TestCase):

    def setUp(self) -> None:
        data = [0,3] # set up 3 product types
        all_warehouses = [] # set up list of all warehouses

    def test_stock_calc(self):
        print("test stock calc")
        # create an order of 1 1 1 products
        products = [Product(0,10),Product(1,15),Product(2,20)]
        orders = []
        orders.append(Order(0,[0,5],0,products,True))

        #create a warehouse with too much of product 2 and not enough of product 1
        example_warehouse = Warehouse(0,[0,0],[1,0,2],orders)

        example_warehouse.stockCalculation()
        print("Warehouse request:", example_warehouse.request)
        print("Warehouse excess:", example_warehouse.excess)

        # numpy array testing
        self.assertIsNone(np.testing.assert_array_equal(example_warehouse.request, np.array([0,1,0])))
        self.assertIsNone(np.testing.assert_array_equal(example_warehouse.excess, np.array([0,0,1])))

    def test_calculate_redistribution(self):
        print("test calculate redist")
        product_data = {}
        product_data.update({0:10})
        product_data.update({1:20})
        product_data.update({2:30})

        products = [Product(0,10),Product(1,20),Product(2,30)]
        orders = []
        orders.append(Order(0,[0,5],0,products,True))

        #create a warehouse with too much of product 2 and not enough of product 1
        example_warehouse = Warehouse(0,[10,10],[1,0,2],orders)

        products2 = [Product(2,30)]
        orders2 = []
        orders2.append(Order(1,[0,5],0,products2,True))
        example_warehouse2 = Warehouse(1,[20,20],[0,1,0],orders2)
        whlist=[example_warehouse,example_warehouse2]
        example_warehouse.stockCalculation()
        example_warehouse2.stockCalculation()


        red = calculate_redistribution(whlist,product_data)
        # definitely something strange going on here, need to check this redistribution data.
        # Testing seems to be saying that the redistribution is going the wrong way.
        print(red)


# %%

if __name__ == "__main__":
    unittest.main()
    





# %%
