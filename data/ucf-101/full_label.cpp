#include <iostream>
#include <map> 
#include <iterator> 
#include <string> 
#include <fstream>

using namespace std;

int main()
{
    ifstream infile1("trainlist01.txt");
    ifstream infile2("trainlist02.txt");
    ifstream infile3("trainlist03.txt");

    ifstream infile_test1("testlist01.txt");
    ifstream infile_test2("testlist02.txt");
    ifstream infile_test3("testlist03.txt");

    ofstream onfile_test1("testlist01_.txt");
    ofstream onfile_test2("testlist02_.txt");
    ofstream onfile_test3("testlist03_.txt");

    map<string, int> my_map; 

    string filename;
    int label;

    while (infile1 >> filename >> label) 
    {
        my_map.insert(make_pair<string,int>(filename, label)); 
    }
    while (infile2 >> filename >> label) 
    {
        my_map.insert(make_pair<string,int>(filename, label)); 
    }
    while (infile3 >> filename >> label) 
    {
        my_map.insert(make_pair<string,int>(filename, label)); 
    }
    cout<<"number: "<<my_map.size()<<endl;

    while (infile_test1 >> filename) 
    {
        onfile_test1 << filename << " " << my_map[filename] << endl;
    }
    while (infile_test2 >> filename) 
    {
        onfile_test2 << filename << " " << my_map[filename] << endl;
    }
    while (infile_test3 >> filename) 
    {
        onfile_test3 << filename << " " << my_map[filename] << endl;
    }
    infile1.close();
    infile2.close();
    infile3.close();

    infile_test1.close();
    infile_test2.close();
    infile_test3.close();

    onfile_test1.close();
    onfile_test2.close();
    onfile_test3.close();
}
