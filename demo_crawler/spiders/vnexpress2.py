import json
import scrapy
from datetime import datetime

OUTPUT_FILENAME = 'output/vnexpress/vnexpress2_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

class VnexpressSpider(scrapy.Spider):
    name = 'vnexpress2'
    start_urls = ['https://vnexpress.net/cuoc-song-nguoi-dan-trong-vung-phong-toa-4140006.html']

    def parse(self, response):
        print('Crawling from:', response.url)
        data = {
            'link': response.url,
            'title': response.css('h1.title-detail::text').get(),
            'description': response.css('p.description::text').get(),

            'content': '\n'.join([
                ''.join(c.css('*::text').getall())
                    for c in response.css('article.fck_detail p.Normal')
            ]),

            'category': response.css('meta[itemprop="articleSection"]::attr("content")').get(),
            'pub_date': float(response.css('meta[name="its_publication"]::attr("content")').get()),
            'keywords': [
                k.strip() for k in response.css('meta[name="keywords"]::attr("content")').get().split(',')
            ],
            'tags': [
                k.strip() for k in response.css('meta[name="its_tag"]::attr("content")').get().split(',')
            ],
        }

        with open(OUTPUT_FILENAME, 'a', encoding='utf8') as f:
            f.write(json.dumps(data, ensure_ascii=False))
            f.write('\n')
            print('SUCCESS:', response.url)

        for href in response.css('a::attr(href)').getall():
            yield response.follow(href, callback=self.parse)
