import json
import re

import scrapy
from datetime import datetime

OUTPUT_FILENAME = 'output/sosanhgia/sosanhgia{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
STORE_SUMMARY_REGEX = re.compile(r'(?:\D+(\d+)\D+(\d+)\D+(\d+(?:\.\d+)*)\D+(\d+(?:\.\d+)*).*|\D+(\d+)\D+(\d+(?:\.\d+)*).*)')
NOT_NUMBER_REGEX = re.compile(r'\D+')


class VnexpressSpider(scrapy.Spider):
    name = 'sosanhgia'
    allowed_domains = ['sosanhgia.com']
    start_urls = ['https://www.sosanhgia.com']
    CRAWLED_COUNT = 0

    def parse(self, response):
        if response.status == 200 and response.css('body::attr("id")').get() == 'product-detail':
            print('Crawling from:', response.url)
            data = {
                'link': response.url,
                'category': response.css('div.breadcrumb-wrapper > ul > li > a > span::text').getall(),
                'name': response.css('div.product-info-container > a > h1::text').get(),
                'img_url': response.css('div.product-slide-container > div.big-image > img::attr("data-src")').get(),
                'brand': response.css('div.product-info-container > div.brand > a::text').get(),
                'short_desc': '\n'.join(response.css('div.product-info-container > div.product-short-desc p::text').getall()),

                'priority_price': response.css('div.product-info-container > div.priority-store span.store-price.product-price::text').get(),
                'priority_store': response.css('div.product-info-container > div.priority-store img::attr("title")').get(),
            }

            stores_summrary = ' '.join(t.strip() for t in response.css('div.product-info-container > div.stores-summrary ::text').getall())
            matched = STORE_SUMMARY_REGEX.match(stores_summrary.strip())
            if matched:
                ret = matched.groups()
                if ret[0] is not None:
                    data['num_of_product'], data['num_of_store'] = int(ret[0]), int(ret[1])
                    data['lowest_price'], data['highest_price'] = ret[2], ret[3]
                else:
                    data['num_of_product'], data['num_of_store'] = int(ret[4]), 1
                    data['lowest_price'], data['highest_price'] = ret[5], '0'
            else:
                data['num_of_product'], data['num_of_store'] = None, None
                data['lowest_price'], data['highest_price'] = None, None

            with open(OUTPUT_FILENAME, 'a', encoding='utf8') as f:
                f.write(json.dumps(data, ensure_ascii=False))
                f.write('\n')
                self.CRAWLED_COUNT += 1
                self.crawler.stats.set_value('CRAWLED_COUNT', self.CRAWLED_COUNT)
                print('SUCCESS:', response.url)

        yield from response.follow_all(css='a[href^="https://www.sosanhgia.com"]::attr(href), a[href^="/"]::attr(href)', callback=self.parse)
