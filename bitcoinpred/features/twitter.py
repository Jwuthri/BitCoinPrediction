# -*- coding: utf-8 -*-"""@author: JulienWuthrich"""import jsonimport reimport datetimefrom http import cookiejarfrom urllib import parse, requestfrom pyquery import PyQueryfrom bitcoinpred.features.settings import twitter_urlclass TweetCollector(object):    """Module to collect tweets"""    def __init__(self, query, count=1, since="2017-12-11T23:50:00", until="2017-12-12T00:00:00"):        """Initialize a `TweetCollector` object.            Args:                query (str): the query you looking for                count (int): number of tweets to collect                since (str): from a date                until (str): until a date        """        self.query = query        self.count = count        self.since = since        self.until = until    def json_request(self, cursor_position):        """Make a request."""        furl = ""        furl += " since:" + self.since        furl += " until:" + self.until        furl += " " + self.query        # furl = " from: {} until: {} {}".format(self.since, self.until, self.query)        url = twitter_url % (parse.quote(furl), "", cursor_position)        headers = [            ('Host', "twitter.com"),            ('User-Agent', "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"),            ('Accept', "application/json, text/javascript, /; q=0.01"),            ('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),            ('X-Requested-With', "XMLHttpRequest"),            ('Referer', url),            ('Connection', "keep-alive")        ]        opener = request.build_opener(request.HTTPCookieProcessor(cookiejar.CookieJar()))        opener.addheaders = headers        response = opener.open(url)        response = response.read()        return json.loads(response.decode())    def collect(self):        """Collect the tweets.            Return:                ltweets (list): list of tweets        """        results = []        results_aux = []        alive = True        cursor_position = ""        while alive:            data = self.json_request(cursor_position)            if len(data["items_html"].strip()) == 0:                break            cursor_position = data["min_position"]            scraped = PyQuery(data["items_html"])            scraped.remove("div.withheld-tweet")            tweets = scraped("div.js-stream-tweet")            if len(tweets) == 0:                break            for tweet_http in tweets:                urls = []                tweet_query = PyQuery(tweet_http)                username = tweet_query("span.username.js-action-profile-name b").text()                txt = re.sub(r"\s+", " ", tweet_query("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'))                retweets = int(tweet_query("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""))                date = int(tweet_query("small.time span.js-short-timestamp").attr("data-time"))                id = tweet_query.attr("data-tweet-id")                user_id = int(tweet_query("a.js-user-profile-link").attr("data-user-id"))                geo_span = tweet_query('span.Tweet-geo')                geo = ''                if len(geo_span) > 0:                    geo = geo_span.attr('title')                for link in tweet_query("a"):                    try:                        urls.append((link.attrib["data-expanded-url"]))                    except KeyError:                        pass                tweet = Tweet()                tweet.id = id                tweet.username = username                tweet.text = txt                tweet.date = datetime.datetime.fromtimestamp(date)                tweet.formatted_date = datetime.datetime.fromtimestamp(date).strftime("%a %b %d %X +0000 %Y")                tweet.retweets = retweets                tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))                tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(tweet.text))                tweet.geo = geo                tweet.urls = ",".join(urls)                tweet.author_id = user_id                results.append(tweet)                results_aux.append(tweet)                if len(results) >= self.count > 0:                    alive = False        return resultsclass Tweet(object):    def __init__(self):        passif __name__ == '__main__':    res = TweetCollector(query="bitcoin, price, crypto, blockchain", count=1, since="2017-12-11T23:50:00", until="2017-12-12T00:00:00").collect()    print(len(res))    print(res[0].text, res[0].date)