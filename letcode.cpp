// letcode.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <direct.h>
#include <iomanip>
#include <cmath>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <algorithm>
#include <sys/timeb.h>
#include <sys/types.h>        /*  socket types              */
#pragma comment (lib, "Ws2_32.lib")

using namespace std;

/* 1. Two Sum */
vector<int> twoSum(vector<int>& nums, int target) {
	vector<int> res;
	unordered_map<int, int> imap;
    
	for (int i = 0; i < nums.size(); ++i) {
        auto it = imap.find(target - nums[i]);
        
		if (it != imap.end()) 
		{
			res.push_back(i);
			res.push_back(it->second);
			return res;
		}
            
		imap[nums[i]] = i;
	}

	return res;
}

/* 2. Add Two Numbers*/
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *p1, *p2, *cur = NULL, *pre = NULL, *ret = NULL;
        int val = 0, inc = 0;
        unsigned int sum = 0;
        
        p1 = l1;
        p2 = l2;
        
        ret = pre = new ListNode(0);
        while(true)
        {          
            if(p1 || p2 || inc)
            {
                val = ((p1 != NULL)?p1->val:0) + ((p2 != NULL)?p2->val:0) + inc;
            }
            else
            {
                break;
            }
            
            inc = val/10;
            val = val - inc*10;
            
            cur = new ListNode(val);

            pre->next = cur;
            pre = cur;
            
            if(p1 != NULL) p1 = p1->next;
            if(p2 != NULL) p2 = p2->next;
        }
        
        cur = ret;
        ret = cur->next;
        delete cur;
        
        return ret; 
}

/* 3. Longest Substring Without Repeating Characters */
int lengthOfLongestSubstring(string s) {
	int bit[256] = {-1}; 
	int i, last = -1, longest = 0;

	memset(bit, -1, sizeof(bit));
	for(i = 0; i < s.size(); i++)
    {
		if(bit[s[i]] != -1) last = (last > bit[s[i]])?last:bit[s[i]];

		bit[s[i]] = i;
		longest = (i - last) > longest? (i - last):longest;
	}

	return longest;
}

/* 4. Median of Two Sorted Arrays*/
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
       int idx = 0, total, median, i = 0, j = 0, val1 = 0, val2 = 0;
        
        total = nums1.size() + nums2.size();
        median = (total + 1)/2;
        
        i = nums1.size();
        j = nums2.size();
        
        if(i == 0 || j == 0)
        {
            (i == 0)?val1 = nums2[j-1]:val1 = nums1[i-1];
        }
        
        while(true)
        {
            if(i && j)
            {
                (nums1[i-1] >= nums2[j-1])? (val2 = val1, val1 = nums1[i-1], i--): (val2 = val1, val1 = nums2[j-1], j--);
            }
            else if(i)
            {
                val2 = val1;
                val1 = nums1[i-1];
                i--;
            }
            else
            {
                val2 = val1;
                val1 = nums2[j-1];
                j--;
            }
            
            idx++;
            if((idx == (median + 1)) || (idx >= total))  break;
        }
        
        return (total % 2)?(val2*1.0):(((val1+val2)*1.0)/2);
}

/* 5. Longest Palindromic Substring */
string longestPalindrome(string s) {
		const int size = s.size();
		int **dp = new int*[size];
        int i, j, l = 0, r = 0, longest = -1;

        for(i = 0; i < size; i++)
			dp[i] = new int[size];

		for(i = 0; i < size; i++)
		{
			for(j = 0; j <size; j++)
				dp[i][j] = 0;
		}

        for(i = 0; i < size; i++)
        {
            for(j = 0; j <= i; j++)
            {
                dp[j][i] = ((s[j] == s[i]) && ((i - j) < 2 || dp[j+1][i-1]));
                if(dp[j][i] && longest < (i - j + 1))
                {
                    longest = i - j + 1;
                    l = j;
                    r = i;
                }
            }
        }
        
        return s.substr(l, r - l + 1);
}

/* 6. ZigZag Conversion */
string convert(string s, int nRows) {
    if (nRows <= 1)
        return s;

    const int len = (int)s.length();
    string *str = new string[nRows];

    int row = 0, step = 1;
    for (int i = 0; i < len; ++i)
    {
        str[row].push_back(s[i]);

        if (row == 0)
            step = 1;
        else if (row == nRows - 1)
            step = -1;

        row += step;
    }

    s.clear();
    for (int j = 0; j < nRows; ++j)
    {
        s.append(str[j]);
    }

    delete[] str;
    return s;
}

/* 7. Reverse Integer */
int reverse(int x) {
    long answer = 0;
    while (x != 0) {
       answer = answer * 10 + x % 10;
       if (answer > INT_MAX || answer < INT_MIN) return 0;
       x /= 10;
	}
    
    return (int)answer;
}

/* 8. String to Integer (atoi) */
int myAtoi(string str) {
        long result = 0;
        int indicator = 1;
        for(int i = 0; i<str.size();)
        {
            i = str.find_first_not_of(' ');
            if(str[i] == '-' || str[i] == '+')
                indicator = (str[i++] == '-')? -1 : 1;
            
            while('0'<= str[i] && str[i] <= '9') 
            {
                result = result*10 + (str[i++]-'0');
                if(result*indicator >= INT_MAX) return INT_MAX;
                if(result*indicator <= INT_MIN) return INT_MIN;                
            }
            
            return result*indicator;
        }
        
        return result;
}

/* 9. Palindrome Number */
bool isPalindrome(int x) {
        if(x < 0) return false;
        if(x < 10) return true;
        if(x % 10 == 0) return false;
        
        int rev = 0;
        
        while(rev < x)
        {
            rev = rev*10 + x%10;
            x = x/10;
        }

        return (rev == x || rev/10 == x);
}

/* 11. Container With Most Water */
int maxArea(vector<int>& height) {
        int maxarea = 0, i = 0, j = height.size() - 1;
        
        while(i < j)
        {
            int h = min(height[i], height[j]);
            maxarea = max(maxarea, h * (j - i));
            while(height[i] <= h && i < j) i++;
            while(height[j] <= h && i < j) j--;
        }
        
        return maxarea;
}

/* 14. Longest Common Prefix*/
class TrieNode{
public:
	int count;
	char word;
	vector<TrieNode> child;
	TrieNode():word('@'), count(0) { }
};

class Trie
{
public:
	TrieNode *root;
    unsigned int obj_c;

public:
	Trie():obj_c(0) { root = new TrieNode(); }
	~Trie() {delete root;}

    void InsertTrie(const string word)
    {
        TrieNode* pLoc = root;
        obj_c++;
        
	    for(int i = 0; i < word.size(); i++)
	    {
		    bool exist = false;

		    for(unsigned int j = 0; j < pLoc->child.size(); j++)
		    {
			    if(pLoc->child[j].word == word[i])
			    {
				    pLoc->child[j].count++;
				    pLoc = &(pLoc->child[j]);
				    exist = true;
				    break;
			    }
		    }

		    if(!exist)
		    {
			    TrieNode temp;
			    temp.word = word[i];
			    temp.count++;
			    pLoc->child.push_back(temp);
			    pLoc = &pLoc->child.back();
		    }
	    }
    }
    
    string LongestCommon()
    {
	    string common = "";

	    if(root->child.size() == 1)
	    {
		    TrieNode *pLoc = &root->child[0];
		    while(pLoc != NULL)
		    {
                if(pLoc->count != this->obj_c)
				    break;
			    common.append(&pLoc->word, 1);
			    if(pLoc->child.size() != 1)
				    break;
			    pLoc = &pLoc->child[0];
		    }
	    }

	    return common;
    }
};

string longestCommonPrefix(vector<string>& strs) {
        Trie T;
        
        for(unsigned int i = 0; i < strs.size(); i++)
        {
            T.InsertTrie(strs[i]);
        }
        
        return T.LongestCommon();
}

/* 15. 3Sum */
vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> result;
	int len = int(nums.size() - 1);

	if(nums.size() < 3)
        return result;

	sort(nums.begin(), nums.end());
	for(int li = 0; li < len; li++)
	{
		int mi, ri;
		mi = li + 1;
		ri = len;

		if(nums[li] > 0 || nums[ri] < 0) break;
		if(li > 0 && nums[li] == nums[li-1]) continue;

		while(mi < ri)
		{
			int sum = nums[li] + nums[mi] + nums[ri];

			if(sum == 0)
			{
				/* we got a sulotion */
				vector<int> res;
				int mi_v = nums[mi];
				int ri_v = nums[ri];

				res.push_back(nums[li]);
				res.push_back(nums[mi]);
				res.push_back(nums[ri]);
				result.push_back(res);
				mi++;
				ri--;
				while(mi_v == nums[mi] && mi < ri) mi++;
				while(ri_v == nums[ri] && ri > mi) ri--;
			}
			else
			{
				sum > 0 ? ri-- : mi++;
			}
		}
	}

	return result;
}

/* 16. 3Sum Closet*/
int threeSumClosest(vector<int>& nums, int target)
{
	int res = 0, dis = INT_MAX;
	int len = int(nums.size() - 1);

	sort(nums.begin(), nums.end());
	for(int li = 0; li < len; li++)
	{
		int mi = li + 1, ri = len;
		while(mi < ri)
		{
			int sum = nums[li] + nums[mi] + nums[ri];
			int t_dis = abs(sum - target);
			if(t_dis < dis)
			{
				dis = t_dis;
				res = sum;
			}

			if(sum == target)
			{
				return sum;
			}
			else 
			{
				sum > target ? ri-- : mi++;
			}
		}
	}

	return res;
}

/* 378. Kth Smallest Element in a Sorted Matrix */
class KTOP_MAXHEAP
{
private:
	int _size;
	int max_size;
	priority_queue<int> pq;

public:
	KTOP_MAXHEAP(int K):_size(0) {max_size = K;}
	void show_queue();
	void push_queue(int v);
	int top() {return pq.top();}
};

void KTOP_MAXHEAP::show_queue()
{
	while(!pq.empty())
	{
		cout<<pq.top()<<" ";
		pq.pop();
	}
	cout<<endl;
}

void KTOP_MAXHEAP::push_queue(int v)
{
	if(pq.size() < max_size){
		pq.push(v);
	}else{
		if(v < pq.top())
		{
			pq.pop();
			pq.push(v);
		}
	}
}

int kthSmallest(vector<vector<int>>& matrix, int k) {
	KTOP_MAXHEAP k_max(k);
	int i,j,v = 0;
	for(i = 0 ; i < matrix.size(); i++)
	{
		for(j = 0; j < matrix[i].size(); j++)
		{
			k_max.push_queue(matrix[i][j]);
		}
	}
	v = k_max.top();
	cout << "Min " << k << "th value is: " << v << endl;
	k_max.show_queue();
	return v;
/* another simple using priority_queue
        priority_queue<int> q;
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < matrix[i].size(); ++j) {
                q.emplace(matrix[i][j]);
                if (q.size() > k) q.pop();
            }
        }
        return q.top();
*/
}

/* 821. Shortest Distance to a Character */
void updateDist(vector<int> &dist, int le, int re, int side)
{
	if(side == 1){
		for(int i = 0; i < re; i++)
		{
			dist[i] = re - le - i;
		}
	}
	else if(side == 2){
		for(int i = le; i <= re; i++)
		{
			dist[i-1] = (re -le) - (re - i); 
		}
	}
	else{
		for(int i = le; i <= re; i++)
		{
			dist[i-1] = min(i - le, re - i);
		}
	}
}
vector<int> shortestToChar(string S, char C) {
	vector<int> e_idxs;
	vector<int> dist;
	int i;

	for(i = 0; i < S.length(); i++)
	{
		dist.push_back(0);
		if(S[i] == C) e_idxs.push_back(i+1);
	}

	if(e_idxs[0] != 1) updateDist(dist, 1, e_idxs[0], 1);
	if(e_idxs[e_idxs.size()-1] != dist.size()) updateDist(dist, e_idxs[e_idxs.size()-1], dist.size(), 2);

	for(i = 0; i < (e_idxs.size()-1); i++)
	{
		updateDist(dist, e_idxs[i], e_idxs[i+1], 0);
	}

	return dist;
}

/* 822. Card Flipping Game */
int flipgame(vector<int>& fronts, vector<int>& backs) {
	int small_good = 2001, back = 0;
	for(int i = 0; i < fronts.size(); i++)
	{
		int j;
		if(fronts[i] == backs[i]) continue;
		for(j = 0; j < fronts.size(); j++)
		{
			if(j == i) continue;
			if(fronts[j] == fronts[i]) break;
		}
		if(j == fronts.size() && (fronts[i] < small_good)) small_good = fronts[i];
	}
	return (small_good < 2001)?small_good:0;
}

int main(int argc, char* argv[])
{
	int arry[] =  {1,2,4,4,7};
	int arryb[] = {1,3,4,1,3};

	vector<int> fronts(arry, arry+5);
	vector<int> backs(arryb, arryb+5);
	printf("result %d\n", flipgame(fronts, backs));
	system("pause");

	return 0;
}

