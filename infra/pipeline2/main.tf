provider "azurerm" {
  features {}
  subscription_id = "ab16a6e7-fa55-4581-b4dd-5602b818cd8e"
  use_cli         = true
}

variable "ssh_public_key" {
  type        = string
  description = "SSH Public Key for the VM"
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH into VM"
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_ip_for_tf_serving" {
  description = "CIDR allowed to access TF Serving"
  type        = string
  default     = "0.0.0.0/0"
}

# Reuse resource group created by pipeline1
data "azurerm_resource_group" "example" {
  name = "test-vm-groupterraf"
}

# Reuse existing virtual network
data "azurerm_virtual_network" "example" {
  name                = "test-vnet"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reuse existing subnet
data "azurerm_subnet" "example" {
  name                 = "test-subnet"
  virtual_network_name = data.azurerm_virtual_network.example.name
  resource_group_name  = data.azurerm_resource_group.example.name
}

# Reuse existing network security group
data "azurerm_network_security_group" "example" {
  name                = "test-nsg"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reuse existing public IP
data "azurerm_public_ip" "example" {
  name                = "myPublicIP"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reuse existing NIC
data "azurerm_network_interface" "example" {
  name                = "test-nic"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Re-attach NSG to NIC to be safe
resource "azurerm_network_interface_security_group_association" "example" {
  network_interface_id      = data.azurerm_network_interface.example.id
  network_security_group_id = data.azurerm_network_security_group.example.id
}

# Create only VM
resource "azurerm_linux_virtual_machine" "example" {
  name                = "test-vm"
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

# Output reused IP address
output "public_ip" {
  value       = data.azurerm_public_ip.example.ip_address
  description = "Public IP of the Azure VM"
  sensitive   = false
}
