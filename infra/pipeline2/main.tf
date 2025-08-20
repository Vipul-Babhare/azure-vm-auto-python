provider "azurerm" {
  features {}
  subscription_id = "ab16a6e7-fa55-4581-b4dd-5602b818cd8e"
  use_cli         = true
}

variable "ssh_public_key" {
  type        = string
  description = "SSH Public Key for the temporary build VM"
}

variable "rg_name" {
  type        = string
  description = "Resource Group Name"
}

variable "vnet_name" {
  type        = string
  description = "Virtual Network Name"
}

variable "subnet_name" {
  type        = string
  description = "Subnet Name"
}

variable "nsg_name" {
  type        = string
  description = "Network Security Group Name"
}

# Unique suffix per pipeline run
resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

# --- Reference existing infrastructure from provision pipeline ---
data "azurerm_resource_group" "rg" {
  name = var.rg_name
}

data "azurerm_virtual_network" "vnet" {
  name                = var.vnet_name
  resource_group_name = data.azurerm_resource_group.rg.name
}

data "azurerm_subnet" "subnet" {
  name                 = var.subnet_name
  virtual_network_name = data.azurerm_virtual_network.vnet.name
  resource_group_name  = data.azurerm_resource_group.rg.name
}

data "azurerm_network_security_group" "nsg" {
  name                = var.nsg_name
  resource_group_name = data.azurerm_resource_group.rg.name
}

# --- Create temporary resources for this build run ---
resource "azurerm_public_ip" "pip_temp" {
  name                = "tempPublicIP-${random_string.suffix.result}"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
  domain_name_label   = "tempvm-${random_string.suffix.result}"
}

resource "azurerm_network_interface" "nic_temp" {
  name                = "temp-nic-${random_string.suffix.result}"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = data.azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.pip_temp.id
  }
}

resource "azurerm_network_interface_security_group_association" "nsg_assoc_temp" {
  network_interface_id      = azurerm_network_interface.nic_temp.id
  network_security_group_id = data.azurerm_network_security_group.nsg.id
}

resource "azurerm_linux_virtual_machine" "temp_vm" {
  name                = "temp-vm-${random_string.suffix.result}"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  size                = "Standard_B1s"
  admin_username      = "azureuser"
  network_interface_ids = [
    azurerm_network_interface.nic_temp.id,
  ]

  admin_ssh_key {
    username   = "azureuser"
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }
}

# Output for GitHub Actions to grab and use
output "public_ip" {
  value       = azurerm_public_ip.pip_temp.ip_address
  description = "Public IP of the temporary build VM"
}
