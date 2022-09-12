# Research Project Website Template
This is a template for research project website. I assume that we are using jekyll for website generation by default.

## Put into your own jekyll based website
1. Create project folder
```shell
	mkdir _projects
```
1. Clone / copy the whole repo into `_projects` repo

1. Modify `_config.yml`:

``` yaml
collections:
	projects:
		output: true
```


## Color
We will use the color pattern following UT color (Burnt Orange and
Charcoal)[https://sites.utexas.edu/cofa-communications/college-brand/color/]: 
- Burnt Orange: #bf5700
- Charcoal: #333f48


## Test the website locally
1. Create a folder

1. Put the `_projects_` folder under the created one.

1. Generate the jekylll configs

``` shell
bundle init
bundle config set --local path 'vendor/bundle'
bundle add jekyll
bundle exec jekyll new --force --skip-bundle .
```

1. Run `bundle exec jekyll serve`

1. Go to `http://127.0.0.1:4000/projects/project-website-template/` (Default)
