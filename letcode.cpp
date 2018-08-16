// letcode.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <set>
#include <unordered_set>
#include <map>
#include <cstdint>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <direct.h>
#include <iomanip>
#include <cmath>
#include <string>
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

/* 830. Positions of Large Groups */
    int recurse(string &S, int s_idx)
    {
        if(s_idx == (S.length()-1))
            return s_idx; 
            
        if(S[s_idx+1] == S[s_idx])
            return recurse(S, s_idx+1);
        else
            return s_idx;
    }
    
    vector<vector<int>> largeGroupPositions(string S) {
        vector<vector<int>> result;
        
		if(S.length() > 3)
		{
			for(int s_idx = 0; s_idx < S.length();)
			{
				int e_idx = recurse(S, s_idx);
				if((e_idx - s_idx) >= 2)
				{
					vector<int> V;
					V.push_back(s_idx);
					V.push_back(e_idx);
					result.push_back(V);
				}
				s_idx = e_idx + 1;
			}
		}
   
        return result;
}

/* 845. Longest Mountain in Array */
int longestMountain(vector<int>& A) {
	int ret = 0;

	if(A.size() < 3) return ret;
	for(int i = 1; i < (A.size()-1); i++)
	{
		int li, ri, loop = 1;

		if(A[i] > A[i-1] && A[i] > A[i+1])
		{
			li = i; ri = i;
		}
		else
		{
			continue;
		}

		while(loop)
		{
			loop = 0;
			if(li > 0 && A[li] > A[li-1])
			{
				li--; loop += 1;
			}
			if(ri < (A.size()-1) && A[ri] > A[ri+1])
			{
				ri++; loop += 1;
			}
		}

		if(ret < (ri - li + 1) && (ri - li + 1) >= 3) ret = ri - li + 1;
	}

	return ret;
}

/* 139. Word Break */
bool wordBreak(string s, vector<string>& wordDict) {
	int min_word = INT_MAX;
	set<string> DictSet;

	for(vector<string>::iterator word = wordDict.begin(); word != wordDict.end(); ++word)
	{
		DictSet.insert(*word);
		min_word = min(min_word, (int)(*word).size());
	}

	if(s.size() < min_word)
		return false;

	vector<bool> Break(s.size(), false);
	for(int i = min_word-1; i < s.size(); i+=min_word)
	{
		for( int j = i - 1; j >= -1; j--)
		{
			if(j == -1 || Break[j])
			{
				if(DictSet.find(s.substr(j+1, i-j)) != DictSet.end())
				{
					Break[i] = true;
					break;
				}
			}
		}
	}

	return Break[s.size()-1];
}

/* 647. Palindromic Substrings */
bool isPalindromic(string s)
{
	int li, ri;
	/* cout << s << endl; */
	if(s.size() == 1) return true;
	li = 0;
	ri = s.size() - 1;
	while(li <= ri)
	{
		if(s[li] == s[ri])
		{
			li++;ri--;
		}
		else
		{
			return false;
		}
	}
	return true;
}

int countSubstrings(string s) {
        int li, i, j, count = 0;
		/* the substr word count */
		for(i = 1; i <= s.size(); i++)
		{
			for(li = 0; li <= s.size() - i; li++)
			{
				if(isPalindromic(s.substr(li, i))) count++;
			}
		}

	return count;
}

/* 617. Merge Two Binary Trees */

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
	if(!t1 && !t2) return nullptr;
	if(!t1) return t2;
	else if(!t2) return t1;
	else
	{
		TreeNode *newnode = new TreeNode(t1->val + t2->val);
		newnode->left = mergeTrees(t1->left, t2->left);
		newnode->right = mergeTrees(t1->right, t2->right);
		return newnode;
	}
}

/* 581. Shortest Unsorted Continuous Subarray */
int findUnsortedSubarray(vector<int>& nums) {
        int start = -1, end = -1,  max = INT_MIN;
        for(int i=0; i<nums.size();i++){
            if(max>nums[i]){
                if(start == -1)start = i-1;
                while(start-1>=0 && nums[start-1]>nums[i])start--;
                end = i+1;
            }
            else max = nums[i];
        }
        return end - start;   
}

/* 146. LRU Cache */
class LRUCache {
public:
    LRUCache(int capacity) {
        max_size = capacity;
    }
    
    int get(int key) {
		int value = -1;
		if(LRUMap.find(key) != LRUMap.end())
		{
			value = LRUMap[key]->second;
			LRUList.erase(LRUMap[key]);
			LRUList.push_front(make_pair(key,value));
			LRUMap[key] = LRUList.begin();
		}

        return value;
    }
    
    void put(int key, int value) {
        if(LRUMap.find(key) != LRUMap.end())
		{
			LRUList.erase(LRUMap[key]);
		}
		else if(LRUMap.size() == max_size)
		{
			LRUMap.erase(LRUList.back().first);
			LRUList.pop_back();
		}

		LRUList.push_front(make_pair(key,value));
		LRUMap[key] = LRUList.begin();
    }
private:
	size_t max_size;
	list<pair<int, int>> LRUList;
	unordered_map<int, list<pair<int,int>>::iterator> LRUMap;
};

/* 44. Wildcard Matching */
bool isMatch(string s, string p) {
	int ps = p.size();
	int ss = s.size();
	/* DB solution to record matching result of s[i] to p[j]*/
	vector<vector<bool>> dp(ss+1, vector<bool>(ps+1, false));
	dp[0][0] = true;

	/* scan from p to match s */
	for(int j = 1; j <= ps; j++)
	{
		if(p[j-1] == '*')
		{
			int i = 0;
			/* any substr if s[0->i] could match p[0->j-1], then p[i][j->end] match */
			while(i <= ss)
			{
				if(dp[i][j-1]) break;
				i++;
			}

			while(i <= ss) dp[i++][j] = true;
		}
		else if(p[j-1] == '?')
		{
			for(int i = 1; i <= ss; i++)
			{
				if(dp[i-1][j-1]) dp[i][j] = true;
			}
		}
		else
		{
			for(int i = 1; i <= ss; i++)
			{
				if((p[j-1] == s[i-1]) && dp[i-1][j-1]) dp[i][j] = 1;
			}
		}
	}

   for (int i=0; i<=ss; i++){
        for (int j=0; j<=ps; j++)
            cout<<dp[i][j]<<" ";
        cout<<endl;
    }
	return dp[ss][ps];
}

/* 10. Regular Expression Matching */
bool isMatch2(string s, string p) {
	int ss = s.size();
	int ps = p.size();

	/* DB solution to record matching result of s[i] to p[j]*/
	vector<vector<bool>> dp(ss+1, vector<bool>(ps+1, false));
	dp[0][0] = true;

	/* s could be empty, so start from index 0. */
	for(int i = 0; i <= ss; i++)
	{
		for(int j = 1; j <= ps;j++)
		{
			if(p[j-1] == '*')
			{
				/* star can repeat zero or repeat many times in case p[j-2] == s[i-1]||p[j-2] == '.' */
				dp[i][j] = dp[i][j-2] || ((i >=1) && (p[j-2] == s[i-1]||p[j-2] == '.') && dp[i-1][j]);
			}
			else
			{
				dp[i][j] = (i >=1) && (p[j-1] == s[i-1]||p[j-1] == '.') && dp[i-1][j-1];
			}
		}
	}

	for (int i=0; i<=ss; i++){
        for (int j=0; j<=ps; j++)
            cout<<dp[i][j]<<" ";
        cout<<endl;
    }
	return dp[ss][ps];
}

/* 140. Word Break II */
vector<string> result;
void WBS(vector<vector<int>> &dp, int s_idx, string s, vector<string> &res)
{
	for(int i = 0; i < dp[s_idx].size(); i++)
	{
		int e_idx = dp[s_idx][i];
		res.push_back(s.substr(s_idx, e_idx-s_idx));

		if(e_idx == s.size())// we get a solution
		{
			string p;
			for(int k = 0; k < (res.size()-1); k++)
			{
				p.append(res[k]);
				p.append(" ");
			}
			p.append(res[res.size()-1]);
			result.push_back(p);
			res.pop_back();
			return;
		}

		WBS(dp, e_idx, s, res);
		res.pop_back();
	}
}

vector<string> wordBreak2(string s, vector<string>& wordDict) {
	int size;
	unordered_set<string> Dict;
	vector<vector<int>> dp(s.size());
	vector<string> res;
	int min_size = INT_MAX, max_size = INT_MIN;

	for(vector<string>::iterator iter = wordDict.begin(); iter != wordDict.end(); iter++)
	{
		int size = (*iter).size();
		Dict.insert(*iter);
		min_size = min(min_size, size);
		max_size = max(max_size, size);
	}
	
	vector<bool> canbreak(s.size(), false);
	for(int i = min_size-1; i < s.size(); i++)
	{
		for(int j = i; j >= -1; j--)
		{
			if(j == -1 || canbreak[j])
			{
				if(Dict.find(s.substr(j+1, i-j)) != Dict.end())
				{
					canbreak[i] = true;
					if( j == -1) dp[0].push_back(i+1);
					else		 dp[j+1].push_back(i+1);
				}
			}

			if((i-j+1) > max_size) break;
		}
	}

	if(canbreak[s.size()-1])
	{
		vector<string> res;
		WBS(dp, 0, s, res);
	}

	return result;
}

/* 454. 4Sum II */
int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
	vector<int> sumAB;
	vector<int> sumCD;
	int i, j, res = 0, rangi, rangj;
    if(A.size() == 0) return res;

	for(i = 0; i < A.size(); i++)
		for(j = 0; j < B.size(); j++)
			sumAB.push_back(A[i]+B[j]);
    sort(sumAB.begin(), sumAB.end());
    rangi = -1*sumAB.back();
    rangj = -1*sumAB.front();

	for(i = 0; i < C.size(); i++)
		for(j = 0; j < D.size(); j++)
    {
            int res = C[i]+D[j];
			if(res >= rangi && res <= rangj) sumCD.push_back(res);
    }
    sort(sumCD.begin(), sumCD.end());
    
    for(i = 0; i < sumAB.size(); i++)
	{
        int delp, target;
		target =  (-1)*sumAB[i];
		delp = sumCD.size();
		for(j = (delp-1); j >=0; j--)
		{
			if(sumCD[j] > target) delp--;
			if(sumCD[j] == target) res++;
			if(sumCD[j] < target) break;
		}
		sumCD.erase(sumCD.begin()+delp, sumCD.end()); 
	}
	return res;
}

int main(int argc, char* argv[])
{
	int A[] = {-268435121,-268434818,-268435384,-268434504,-268434852,-268434847,-268434718,-268434830,-268434644,-268434843,-268435282,-268434756,-268435098,-268435361,-268434888,-268435198,-268434771,-268435001,-268435011,-268434745,-268434892,-268435442,-268434705,-268434932,-268435095,-268434829,-268435166,-268434672,-268434959,-268434886,-268435158,-268435439,-268434665,-268434561,-268435327,-268434478,-268434879,-268434902,-268435027,-268434922,-268435028,-268434950,-268435407,-268434916,-268435326,-268435048,-268435156,-268435043,-268434607,-268434738,-268434944,-268434813,-268435194,-268434793,-268435372,-268434483,-268434761,-268434571,-268434529,-268435314,-268434733,-268435218,-268434590,-268434520,-268435372,-268435180,-268435420,-268435360,-268435316,-268435455,-268434797,-268435008,-268434540,-268434816,-268434485,-268435238,-268434559,-268434589,-268435257,-268434877,-268435455,-268435128,-268435307,-268434919,-268434893,-268435239,-268434788,-268435176,-268434930,-268434582,-268434471,-268435206,-268434561,-268434880,-268435137,-268434504,-268434549,-268434825,-268434587,-268435293};
	int B[] = {-268434497,-268435316,-268435433,-268434992,-268435391,-268434629,-268434902,-268434629,-268434895,-268434527,-268434822,-268435325,-268434634,-268434877,-268434758,-268434758,-268435109,-268435179,-268434736,-268435111,-268435243,-268435191,-268435091,-268435041,-268435231,-268434473,-268434656,-268435379,-268435127,-268434727,-268435302,-268434643,-268435101,-268434879,-268434848,-268434880,-268434817,-268434746,-268435005,-268435018,-268434535,-268435390,-268435430,-268434486,-268434497,-268434652,-268434816,-268435055,-268435296,-268435044,-268434939,-268435267,-268435283,-268434673,-268435298,-268435069,-268435041,-268434525,-268435023,-268435064,-268434969,-268434630,-268435197,-268435049,-268434496,-268434777,-268434928,-268434683,-268434923,-268434699,-268434634,-268434613,-268435438,-268435449,-268434498,-268435183,-268434577,-268434515,-268435100,-268435023,-268435214,-268435358,-268435394,-268434734,-268434636,-268435056,-268434829,-268434962,-268434673,-268434578,-268434850,-268435294,-268434752,-268435410,-268434841,-268434952,-268435111,-268434905,-268434791,-268434507};
	int C[] = {268434624,268434831,268435028,268434547,268435027,268435453,268435101,268435378,268435107,268434991,268435147,268435379,268435435,268435439,268435003,268435440,268435454,268434786,268434521,268434678,268435125,268434685,268434482,268435042,268434486,268434521,268435003,268435086,268434786,268434762,268435384,268434577,268435199,268435020,268434709,268435006,268434820,268434830,268435208,268435448,268435274,268435438,268435082,268435140,268434465,268435324,268434645,268435032,268434567,268435057,268435161,268435246,268434515,268434880,268435432,268434905,268435135,268435436,268434923,268435110,268434607,268435286,268435281,268434717,268435367,268434894,268435280,268434880,268435181,268434971,268435316,268435021,268435208,268434589,268434811,268434774,268434843,268434762,268434650,268435068,268434731,268435192,268434596,268434549,268434535,268435372,268435361,268435224,268435033,268434807,268435003,268434647,268435051,268434464,268435326,268434991,268434571,268435192,268435435,268435181};
	int D[] = {268434917,268435022,268435311,268434898,268435389,268435347,268434735,268435285,268434695,268434794,268435393,268435114,268435063,268434517,268434464,268434766,268435316,268434706,268434962,268435078,268434921,268435070,268434569,268434895,268435446,268435003,268435362,268435421,268435158,268435274,268435391,268435298,268435311,268435295,268435428,268435010,268434884,268435055,268434824,268434877,268434467,268435181,268434797,268435142,268434611,268434719,268435081,268434924,268435340,268434961,268435215,268434583,268435192,268435364,268434925,268435396,268434599,268435330,268435253,268435037,268434642,268435228,268435103,268434736,268435123,268435220,268435394,268435015,268434906,268434478,268434956,268434465,268434840,268434648,268434706,268435382,268435297,268434602,268435264,268435237,268434582,268434686,268434825,268435246,268435234,268435441,268435139,268435357,268434512,268435375,268434873,268435318,268434469,268434750,268435217,268434469,268434872,268435218,268435254,268434547};
	int size = sizeof(A)/sizeof(int);
	vector<int> A1(A, A+size);
	vector<int> B1(B, B+size);
	vector<int> C1(C, C+size);
	vector<int> D1(D, D+size);
	fourSumCount(A1, B1, C1, D1);

    //printf("result: %d\n",isMatch2("aasdfasdfasdfasdfas", "aasdf.*asdf.*asdf.*asdf.*s"));
	system("pause");

	return 0;
}

