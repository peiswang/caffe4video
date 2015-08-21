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

    ofstream onfile_train1("trainlist01_.txt");
    ofstream onfile_train2("trainlist02_.txt");
    ofstream onfile_train3("trainlist03_.txt");

    map<string, int> my_map; 

    string filename;
    int label;

    while (infile1 >> filename >> label) 
    {
        my_map.insert(make_pair<string,int>(filename, label)); 
        onfile_train1 << filename << " " << label - 1 << endl;
    }
    while (infile2 >> filename >> label) 
    {
        my_map.insert(make_pair<string,int>(filename, label)); 
        onfile_train2 << filename << " " << label - 1 << endl;
    }
    while (infile3 >> filename >> label) 
    {
        my_map.insert(make_pair<string,int>(filename, label)); 
        onfile_train3 << filename << " " << label - 1 << endl;
    }
    cout<<"number: "<<my_map.size()<<endl;

    while (infile_test1 >> filename) 
    {
        onfile_test1 << filename << " " << my_map[filename] - 1 << endl;
    }
    while (infile_test2 >> filename) 
    {
        onfile_test2 << filename << " " << my_map[filename] - 1 << endl;
    }
    while (infile_test3 >> filename) 
    {
        onfile_test3 << filename << " " << my_map[filename] - 1 << endl;
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

    onfile_train1.close();
    onfile_train2.close();
    onfile_train3.close();
}
