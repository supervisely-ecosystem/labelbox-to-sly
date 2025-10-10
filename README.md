<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/labelbox-to-sly/assets/79905215/1ababed7-1960-43f4-91f0-c35ec68b34ad"/>

# Convert and copy multiple Labelbox projects into Supervisely at once

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../supervisely-ecosystem/labelbox-to-sly)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/labelbox-to-sly)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/labelbox-to-sly.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/labelbox-to-sly.png)](https://supervisely.com)

</div>

## Overview

This application allows you to copy multiple projects from Labelbox instance to Supervisely instance, you can select which projects should be copied. You can preview the results in the table, which will show URLs to corresdponding projects in Labelbox and Supervisely.<br>

## Preparation

ℹ️ NOTE: There are some limitations on the Labelbox side depending on your subscription plan. You can find more information about it [here](https://docs.labelbox.com/docs/limits).

ℹ️ NOTE: It is allowed to export only Images or Videos projects from Labelbox, so you need to make sure that your projects meet this requirement, otherwise it's impossible to export data from Labelbox.

In order to run the app, you need to obtain `Private API key` to work with Labelbox API. You can refer to [this documentation](https://docs.labelbox.com/reference/create-api-key) to do it.

The API key should looks like this: `qASymt32UTnQV1qABszFasnvscmasissdjfdJJAhjdhajkfANFJjnNCANCancanlCNANCncncNlcnlncnNDJCNclnDNCcnjNDCjcnjcNJDCNjcnjnCJnjcnjcnjNCJNJCjncJnjcJNDCXANXSsfhasjfha3kjkfjas8fsjf8sf8sfaNXSFHFHFJHfhkHFKhfhHFAkljjjajjDJIACIaucasnkadajdkasMADAfasfsfFFFDDSGDFSDdghfghs7d9325tegd1dDDYAD797ydDDd97DdD97d9D79D9dd90ff4ff4`

Now you have two options to use your API key: you can use team files to store an .env file with API key or you can enter the API key directly in the app GUI. Using team files is recommended as it is more convenient and faster, but you can choose the option that is more suitable for you.

### Using team files

You can download an example of the .env file [here](https://github.com/supervisely-ecosystem/labelbox-to-sly/files/13227776/labelbox.env.zip) and edit it without any additional software in any text editor.<br>
ℹ️ NOTE: you need to unzip the file before using it.<br>

1. Create a .env file with the following content:
   `LB_API_KEY=<your Labelbox API key>`
2. Upload the .env file to the team files.
3. Right-click on the .env file, select `Run app` and choose the `Labelbox to Supervisely Migration Tool` app.

The app will be launched with the API key from the .env file and you won't need to enter it manually.
If everything was done correctly, you will see the following message in the app UI:

- ℹ️ Connection settings was loaded from .env file.
- ✅ Successfully connected to `https://app.labelbox.com`.

### Entering credentials manually

1. Launch the app from the Ecosystem.
2. Enter the API key.
3. Press the `Connect to Labelbox` button.

![credentials](https://github.com/supervisely-ecosystem/labelbox-to-sly/assets/79905215/a14ec953-37a1-42b7-9dde-cc73fe5a84d9)<br>

![credentials](https://github.com/supervisely-ecosystem/labelbox-to-sly/assets/79905215/82df4cc9-0b15-4081-9d65-6f508eadffa2)

If everything was done correctly, you will see the following message in the app UI:

- ✅ Successfully connected to `https://app.labelbox.com`.<br>

ℹ️ NOTE: The app will not save your API key, you will need to enter it every time you launch the app. To save your time you can use the team files to store your credentials.

## How To Run

ℹ️ NOTE: In this section, we consider that you have already connected to Labelbox instance and have the necessary permissions to work with it. If you haven't done it yet, please refer to the [Preparation](#Preparation) section.<br>
So, here is the step-by-step guide on how to use the app:

**Step 1:** Select projects to copy<br>
After connecting to the Labelbox instance, list of the projects will be loaded into the widget automatically. You can select which projects you want to copy to Supervisely and then press the `Select projects` button.<br>

![select_projects](https://github.com/supervisely-ecosystem/labelbox-to-sly/assets/79905215/304bd682-829e-4f03-9692-f3487bef2059)

**Step 2:** Take a look on list of projects<br>
After completing the `Step 1️⃣`, the application will retrieve information about the projects from labelbox API and show it in the table. Here you can find the links to the projects in Labelbox, and after copying the projects to Supervisely, links to the projects in Supervisely will be added to the table too.<br>

![projects_table](https://github.com/supervisely-ecosystem/labelbox-to-sly/assets/79905215/72cbbf1f-3881-41e5-95d6-96c193bfe2b6)<br>


**Step 3:** Press the `Copy` button<br>
Now you only need to press the `Copy` button and wait until the copying process is finished. You will see the statuses of the copying process for each project in the table. If any errors occur during the copying process, you will see the error status in the table. When the process is finished, you will see the total number of successfully copied projects and the total number of projects that failed to copy.<br>

![copy_projects](https://github.com/supervisely-ecosystem/labelbox-to-sly/assets/79905215/5b18d76d-cd62-4d92-b19a-c32f080b4e2c)<br>

![finished](https://github.com/supervisely-ecosystem/labelbox-to-sly/assets/79905215/374832b4-7394-4181-bc9d-7d3fbdad2377)<br>

The application will be stopped automatically after the copying process is finished.<br>

ℹ️ The app supports following Labelbox ontology types (geometry types):
- Images project:
    - bounding box
    - polygon
    - segmentation
    - point
    - polyline
- Videos project:
    - bounding box
    - segmentation
    - point
    - polyline

## Acknowledgement

- [Labelbox Python GitHub](https://github.com/Labelbox/labelbox-python) ![GitHub Org's stars](https://img.shields.io/github/stars/Labelbox/labelbox-python?style=social)
- [Labelbox Documentation](https://docs.labelbox.com/)
