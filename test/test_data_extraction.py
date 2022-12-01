from src.data_extraction import data_extraction


def test_data_extractions():
    
    path = '../data/weatherAUS.csv'
    data = data_extraction(path)
    assert data.shape == (145460, 23)
    

