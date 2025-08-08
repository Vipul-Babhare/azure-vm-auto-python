provider "azurerm" {
  features {}
  subscription_id = "ab16a6e7-fa55-4581-b4dd-5602b818cd8e"
  use_cli         = true
}

variable "ssh_public_key" {
  type        = string
  description = "SSH Public Key for the VM"
}

# Reference existing Resource Group
data "azurerm_resource_group" "example" {
  name = "test-vm-groupterraf"
}

# Reference existing Virtual Network
data "azurerm_virtual_network" "example" {
  name                = "test-vnet"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reference existing Subnet
data "azurerm_subnet" "example" {
  name                 = "test-subnet"
  virtual_network_name = data.azurerm_virtual_network.example.name
  resource_group_name  = data.azurerm_resource_group.example.name
}

# Reference existing Network Security Group (already has rules from pipeline 1)
data "azurerm_network_security_group" "example" {
  name                = "test-nsg"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reference existing Public IP
data "azurerm_public_ip" "example" {
  name                = "myPublicIP"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reference existing Network Interface
data "azurerm_network_interface" "example" {
  name                = "test-nic"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Create new VM for pipeline 2
resource "azurerm_linux_virtual_machine" "example" {
  name                = "test-vm-pipeline2"
  resource_group_name = data.azurerm_resource_group.example.name
  location            = data.azurerm_resource_group.example.location
  size                = "Standard_B1s"
  admin_username      = "azureuser"
  network_interface_ids = [
    data.azurerm_network_interface.example.id,
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

output "public_ip" {
  value       = data.azurerm_public_ip.example.ip_address
  description = "Public IP of the Azure VM for pipeline 2"
  sensitive   = false
}
