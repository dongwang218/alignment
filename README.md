# Align and Recover

Use homography and optical flow to align scanned documents, recover the base document.

![alt tag](https://raw.githubusercontent.com/dongwang218/alignment/master/corrected.jpg)

from these documents:
![alt tag](https://raw.githubusercontent.com/dongwang218/alignment/master/collate.jpg)

## Identify the best image
```
python flow.py ../sheetimages/FmE8AlBI5cIoONcCM0Q_rTmsrcGn.jpg \
		../sheetimages/FmF2M5yU-99rgHVgyiQwMkKWmU8O.jpg \
		../sheetimages/FsYORjHsCNCd8uocKSc-GQsra6TQ.jpg \
		../sheetimages/Ftpm9Zak2v7S6tL0i1X5JBuYgn6_.jpg \
		../sheetimages/Fo6u7XKx1SChQbnwZEW8QOTCvMRd.jpg \
		../sheetimages/FoL5RM60CMVGZmNN4tcu33ZImDhf.jpg \
		../sheetimages/FvV_82jIQ8ZwKqvIjnNn98-RWE8V.jpg \
		../sheetimages/FmoFvXUNSFIgWIAeCPfIqlWGnEjy.jpg \
		../sheetimages/FvTY_1B7gnV-ew_-7jzXB94T8vXG.jpg \
		../sheetimages/FqZfihXuOB68LS-Ja7IU4-AQGVfu.jpg \
		../sheetimages/Fswritmb5z17u77lJWTtEr17iA5N.jpg \
		../sheetimages/FivoFl9_agFgCv1kUTzYgjFhvS_S.jpg \
		../sheetimages/FjlnO2byeMnIT1dzujC4VXCEZEWD.jpg \
		../sheetimages/FnW5wZkLkR27ed1ojXBrOw_feOL1.jpg \
		../sheetimages/Foiy6wWxeGmfRSZoBVhAwCSm1gnO.jpg \
		../sheetimages/FkZS1mRMJwCuN6V4N-N6LrGu_hVh.jpg \
		../sheetimages/Foo5uOB8-TaTqYUTqzWSrTZJsHOo.jpg \
		../sheetimages/Fu8KHfM3Sobjrhvi0oG2JGP6p8W6.jpg \
		../sheetimages/FqVP5GHx4hMV3xO_ivkI6VpyS6Xu.jpg \
		../sheetimages/Fn2JBOjeZ3iotoPrYIOxg-yjfUbj.jpg \
		../sheetimages/FvgiCI8gz6b8pSUAdXAjBX8cepkC.jpg \
		../sheetimages/FgBHmax297wJbG6PAnL02_7C4_ob.jpg \
		../sheetimages/FnlnNassJCrclHYeVTY6fI9BBzQF.jpg \
		../sheetimages/FrelXNFfDDnuIsX1A0QjLb3xneB-.jpg \
		../sheetimages/FhLw1h7qhy-1lSa6yJAO_HhbbDVw.jpg \
		../sheetimages/FmHo21T0yEAFJRbwBBPnERl_BVGm.jpg \
		../sheetimages/Fqy32BSEIll96YRvLWq4VyKgS5cw.jpg \
		../sheetimages/FnNk5mL15wspdM-jUne6clH6HQ0X.jpg \
		../sheetimages/Fmv11OQgeHQSmPp8zkGEfUAfBLk9.jpg
```

## Alignment
python align.py --flow smallest_flow.npy ../sheetimages/FmE8AlBI5cIoONcCM0Q_rTmsrcGn.jpg \
		../sheetimages/FmF2M5yU-99rgHVgyiQwMkKWmU8O.jpg \
		../sheetimages/FsYORjHsCNCd8uocKSc-GQsra6TQ.jpg \
		../sheetimages/Ftpm9Zak2v7S6tL0i1X5JBuYgn6_.jpg \
		../sheetimages/Fo6u7XKx1SChQbnwZEW8QOTCvMRd.jpg \
		../sheetimages/FoL5RM60CMVGZmNN4tcu33ZImDhf.jpg \
		../sheetimages/FvV_82jIQ8ZwKqvIjnNn98-RWE8V.jpg \
		../sheetimages/FmoFvXUNSFIgWIAeCPfIqlWGnEjy.jpg \
		../sheetimages/FvTY_1B7gnV-ew_-7jzXB94T8vXG.jpg \
		../sheetimages/FqZfihXuOB68LS-Ja7IU4-AQGVfu.jpg \
		../sheetimages/Fswritmb5z17u77lJWTtEr17iA5N.jpg \
		../sheetimages/FivoFl9_agFgCv1kUTzYgjFhvS_S.jpg \
		../sheetimages/FjlnO2byeMnIT1dzujC4VXCEZEWD.jpg \
		../sheetimages/FnW5wZkLkR27ed1ojXBrOw_feOL1.jpg \
		../sheetimages/Foiy6wWxeGmfRSZoBVhAwCSm1gnO.jpg \
		../sheetimages/FkZS1mRMJwCuN6V4N-N6LrGu_hVh.jpg \
		../sheetimages/Foo5uOB8-TaTqYUTqzWSrTZJsHOo.jpg \
		../sheetimages/Fu8KHfM3Sobjrhvi0oG2JGP6p8W6.jpg \
		../sheetimages/FqVP5GHx4hMV3xO_ivkI6VpyS6Xu.jpg \
		../sheetimages/Fn2JBOjeZ3iotoPrYIOxg-yjfUbj.jpg \
		../sheetimages/FvgiCI8gz6b8pSUAdXAjBX8cepkC.jpg \
		../sheetimages/FgBHmax297wJbG6PAnL02_7C4_ob.jpg \
		../sheetimages/FnlnNassJCrclHYeVTY6fI9BBzQF.jpg \
		../sheetimages/FrelXNFfDDnuIsX1A0QjLb3xneB-.jpg \
		../sheetimages/FhLw1h7qhy-1lSa6yJAO_HhbbDVw.jpg \
		../sheetimages/FmHo21T0yEAFJRbwBBPnERl_BVGm.jpg \
		../sheetimages/Fqy32BSEIll96YRvLWq4VyKgS5cw.jpg \
		../sheetimages/FnNk5mL15wspdM-jUne6clH6HQ0X.jpg \
		../sheetimages/Fmv11OQgeHQSmPp8zkGEfUAfBLk9.jpg